import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 2, in_chans: int = 16, embed_dim: int = 1536):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool = True):
        super().__init__()
        self.is_double = double
        if double:
            self.lin = nn.Linear(dim, 6 * dim, bias=True)
        else:
            self.lin = nn.Linear(dim, 3 * dim, bias=True)

    def forward(self, vec: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.lin(F.silu(vec))[:, None, :]
        return out.chunk(6 if self.is_double else 3, dim=2)

class JointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 24):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_x = nn.Linear(dim, dim * 3, bias=True)
        self.qkv_c = nn.Linear(dim, dim * 3, bias=True)
        
        self.proj_x = nn.Linear(dim, dim, bias=True)
        self.proj_c = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_x, C = x.shape
        _, N_c, _ = c.shape

        qkv_x = self.qkv_x(x).reshape(B, N_x, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_x, k_x, v_x = qkv_x.unbind(0)

        qkv_c = self.qkv_c(c).reshape(B, N_c, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = qkv_c.unbind(0)

        q = torch.cat([q_x, q_c], dim=2)
        k = torch.cat([k_x, k_c], dim=2)
        v = torch.cat([v_x, v_c], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_out = attn @ v

        x_out, c_out = x_out.split([N_x, N_c], dim=2)
        
        x_out = x_out.transpose(1, 2).reshape(B, N_x, C)
        c_out = c_out.transpose(1, 2).reshape(B, N_c, C)

        return self.proj_x(x_out), self.proj_c(c_out)

class MMDiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        self.attn = JointAttention(dim, num_heads=num_heads)
        
        self.norm2_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_x = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim)
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        self.adaLN_modulation = Modulation(dim, double=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor, vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_msa_x, scale_msa_x, gate_msa_x, shift_msa_c, scale_msa_c, gate_msa_c = self.adaLN_modulation(vec)
        
        x_norm = self.norm1_x(x) * (1 + scale_msa_x) + shift_msa_x
        c_norm = self.norm1_c(c) * (1 + scale_msa_c) + shift_msa_c
        
        x_attn, c_attn = self.attn(x_norm, c_norm)
        
        x = x + gate_msa_x * x_attn
        c = c + gate_msa_c * c_attn
        
        x = x + self.mlp_x(self.norm2_x(x)) 
        c = c + self.mlp_c(self.norm2_c(c))
        
        return x, c

class MMDiT(nn.Module):
    def __init__(self, 
                 input_size: int = 64, 
                 patch_size: int = 2, 
                 in_channels: int = 16,
                 hidden_size: int = 1536, 
                 depth: int = 24, 
                 num_heads: int = 24, 
                 text_embed_dim: int = 4096): 
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Linear(2048, hidden_size) 
        self.context_embedder = nn.Linear(text_embed_dim, hidden_size) 
        
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = Modulation(hidden_size, double=False) 
        self.final_layer = nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        c = self.in_channels
        p = self.patch_size
        
        x = x.reshape(x.shape[0], h // p, w // p, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h, w)
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x) + self.pos_embed 
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = self.context_embedder(context)
        
        vec = t_emb + y_emb
        
        for block in self.blocks:
            x, c = block(x, c, vec)
            
        shift, scale, gate = self.final_adaLN(vec)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.final_layer(x)
        
        h = w = int(x.shape[1] ** 0.5) * self.patch_size
        x = self.unpatchify(x, h, w)
        
        return x

class FlowMatchScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift 

    def set_timesteps(self, num_inference_steps: int, device: str = "cuda"):
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps, device=device)
    
    def step(self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor, dt: float) -> torch.Tensor:
        prev_sample = sample + model_output * dt
        return prev_sample

class SD3TextEmbedder(nn.Module):
    def __init__(self, device="cuda", max_length=77):
        super().__init__()
        self.device = device
        
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
        self.t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small").to(device)
        
        self.clip_model.eval().requires_grad_(False)
        self.t5_model.eval().requires_grad_(False)

    def forward(self, prompts: List[str]):
        clip_input = self.clip_tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clip_out = self.clip_model(clip_input.input_ids).pooler_output
            
        y = F.pad(clip_out, (0, 2048 - 768)) 

        t5_input = self.t5_tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            t5_out = self.t5_model(t5_input.input_ids).last_hidden_state
        
        context = F.pad(t5_out, (0, 4096 - 512))
        
        return y, context

class SD3():
    def __init__(self, device="cuda"):
        self.device = device
        self.img_size = 64
        self.text_encoder = SD3TextEmbedder(device)
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", subfolder=None).to(device)
        self.transformer = MMDiT(
            input_size=self.img_size,
            patch_size=2,
            in_channels=4, 
            hidden_size=1152, 
            depth=24,
            num_heads=16,
            text_embed_dim=4096
        ).to(device)
        
        self.scheduler = FlowMatchScheduler()
        
        self.vae.eval().requires_grad_(False)
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.13025

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.13025 
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs.cpu().permute(0, 2, 3, 1).numpy()

    @torch.no_grad()
    def generate(self, prompt: str, steps: int = 28, cfg_scale: float = 7, transformer_override=None):
        transformer = transformer_override if transformer_override is not None else self.transformer
        transformer.eval()
        
        y_cond, c_cond = self.text_encoder([prompt])
        y_uncond, c_uncond = self.text_encoder([""]) 
        
        y = torch.cat([y_uncond, y_cond])
        c = torch.cat([c_uncond, c_cond])
        
        latents = torch.randn((1, 4, self.img_size, self.img_size)).to(self.device)
        
        self.scheduler.set_timesteps(steps, device=self.device)
        
        for i, t in enumerate(self.scheduler.timesteps):
            t_batch = torch.cat([t.unsqueeze(0)] * 2)
            latent_batch = torch.cat([latents] * 2)
            
            noise_pred = transformer(latent_batch, t_batch, y, c)
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            if i < len(self.scheduler.timesteps) - 1:
                dt = self.scheduler.timesteps[i+1] - t
            else:
                dt = 0
                
            latents = self.scheduler.step(noise_pred, t, latents, dt)
            
        return self.decode_latents(latents)