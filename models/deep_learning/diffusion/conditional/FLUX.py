import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class EmbedND(nn.Module):
    def __init__(self, dim: int = 3072, theta: int = 10000, axes_dim: List[int] = [16, 56, 56]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, seq_len: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        head_dim = self.dim // 24 
        target_len = h * w
        dummy_freqs = torch.polar(torch.ones(target_len, head_dim // 2, device=device), 
                                  torch.zeros(target_len, head_dim // 2, device=device))
        return dummy_freqs

class TimeGuidanceEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.time_proj = nn.Linear(256, hidden_size)
        self.guidance_proj = nn.Linear(256, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def embedding(self, t: torch.Tensor, dim: int = 256):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half).float() / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        t_emb = self.embedding(t)
        g_emb = self.embedding(guidance)
        
        vec = self.time_proj(t_emb) + self.guidance_proj(g_emb)
        return self.mlp(vec)

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm1_img = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm1_txt = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        self.q_norm_img = RMSNorm(self.head_dim)
        self.k_norm_img = RMSNorm(self.head_dim)
        self.q_norm_txt = RMSNorm(self.head_dim)
        self.k_norm_txt = RMSNorm(self.head_dim)
        
        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3)
        self.qkv_txt = nn.Linear(hidden_size, hidden_size * 3)
        self.proj_img = nn.Linear(hidden_size, hidden_size)
        self.proj_txt = nn.Linear(hidden_size, hidden_size)
        
        self.norm2_img = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2_txt = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp_img = nn.Sequential(nn.Linear(hidden_size, int(hidden_size*mlp_ratio)), nn.GELU(), nn.Linear(int(hidden_size*mlp_ratio), hidden_size))
        self.mlp_txt = nn.Sequential(nn.Linear(hidden_size, int(hidden_size*mlp_ratio)), nn.GELU(), nn.Linear(int(hidden_size*mlp_ratio), hidden_size))

        self.adaLN = nn.Linear(hidden_size, hidden_size * 6)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        chunks = self.adaLN(F.silu(vec))[:, None, :].chunk(6, dim=2)
        (shift_msa_i, scale_msa_i, gate_msa_i, shift_msa_t, scale_msa_t, gate_msa_t) = chunks
        
        img_norm = self.norm1_img(img) * (1 + scale_msa_i) + shift_msa_i
        txt_norm = self.norm1_txt(txt) * (1 + scale_msa_t) + shift_msa_t
        
        B, N_i, C = img.shape
        _, N_t, _ = txt.shape
        
        qkv_i = self.qkv_img(img_norm).reshape(B, N_i, 3, self.num_heads, self.head_dim)
        qkv_t = self.qkv_txt(txt_norm).reshape(B, N_t, 3, self.num_heads, self.head_dim)
        
        q_i, k_i, v_i = qkv_i.unbind(2)
        q_t, k_t, v_t = qkv_t.unbind(2)
        
        q_i, k_i = self.q_norm_img(q_i), self.k_norm_img(k_i)
        q_t, k_t = self.q_norm_txt(q_t), self.k_norm_txt(k_t)
        
        q_i, k_i = apply_rope(q_i, k_i, pe) 
        
        q = torch.cat([q_t, q_i], dim=1) 
        k = torch.cat([k_t, k_i], dim=1)
        v = torch.cat([v_t, v_i], dim=1)
        
        attn = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
        attn = attn.permute(0, 2, 1, 3).flatten(2) 
        
        txt_attn, img_attn = attn.split([N_t, N_i], dim=1)
        
        img = img + gate_msa_i * self.proj_img(img_attn)
        txt = txt + gate_msa_t * self.proj_txt(txt_attn)
        
        img = img + self.mlp_img(self.norm2_img(img))
        txt = txt + self.mlp_txt(self.norm2_txt(txt))
        
        return img, txt

class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        self.proj_mlp = nn.Linear(hidden_size * 4, hidden_size) 
        self.mlp_act = nn.GELU()
        self.mlp_linear = nn.Linear(hidden_size, hidden_size * 4) 
        
        self.adaLN = nn.Linear(hidden_size, hidden_size * 3) 

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN(F.silu(vec))[:, None, :].chunk(3, dim=2)
        
        x_norm = self.norm(x) * (1 + scale) + shift
        
        qkv = self.qkv_linear(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        B, N, C = q.shape
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, pe) 
        
        attn = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
        attn_out = attn.permute(0, 2, 1, 3).flatten(2)
        
        mlp_hidden = self.mlp_linear(x_norm)
        mlp_out, _ = mlp_hidden.chunk(2, dim=-1) 
        mlp_out = self.mlp_act(mlp_out)
        
        x = x + gate * (attn_out + mlp_out) 
        
        return x

class FluxTransformer(nn.Module):
    def __init__(self, 
                 patch_size: int = 1, 
                 in_channels: int = 64, 
                 hidden_size: int = 3072, 
                 depth_double: int = 19, 
                 depth_single: int = 38, 
                 num_heads: int = 24,
                 guidance_embed: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        self.img_in = nn.Linear(in_channels, hidden_size)
        
        self.txt_pooled_in = nn.Linear(768, hidden_size) 
        self.txt_t5_in = nn.Linear(4096, hidden_size) 
        
        self.time_guidance_embed = TimeGuidanceEmbedder(hidden_size)
        
        self.rope_embed = EmbedND(dim=hidden_size, theta=10000)
        
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads) for _ in range(depth_double)
        ])
        
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads) for _ in range(depth_single)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_proj = nn.Linear(hidden_size, in_channels) # Project về pixel space (patch)

    def forward(self, img: torch.Tensor, t: torch.Tensor, guidance: torch.Tensor, 
                txt_pooled: torch.Tensor, txt_t5: torch.Tensor) -> torch.Tensor:
        
        img = self.img_in(img) 
        txt = self.txt_t5_in(txt_t5) 
        
        vec = self.time_guidance_embed(t, guidance) + self.txt_pooled_in(txt_pooled)
        
        h = w = int(math.sqrt(img.shape[1])) 
        pe = self.rope_embed(img.shape[1], h, w, img.device)
        
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, pe)
            
        x = torch.cat([txt, img], dim=1)
        
        for block in self.single_blocks:
            x = block(x, vec, pe)
            
        txt_out, img_out = x.split([txt.shape[1], img.shape[1]], dim=1)
        
        img_out = self.final_norm(img_out)
        img_out = self.final_proj(img_out)
        
        return img_out

class FluxTextEmbedder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        
        self.clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        self.t5_tok = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
        self.t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small").to(device)
        
        self.clip_model.eval().requires_grad_(False)
        self.t5_model.eval().requires_grad_(False)

    def forward(self, prompts: List[str]):
        clip_in = self.clip_tok(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pooled = self.clip_model(clip_in.input_ids).pooler_output 
        t5_in = self.t5_tok(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            txt_seq = self.t5_model(t5_in.input_ids).last_hidden_state
        
        txt_seq = F.pad(txt_seq, (0, 4096 - 512))
        
        return pooled, txt_seq

class FlowMatchScheduler:
    def __init__(self, shift: float = 3.0):
        self.shift = shift 

    def set_timesteps(self, num_inference_steps: int, device: str = "cuda"):
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps, device=device)
    
    def step(self, model_output, t, sample, dt):
        return sample + model_output * dt

class FLUX:
    def __init__(self, device="cuda"):
        self.device = device
        self.img_size = 64 
        
        self.text_encoder = FluxTextEmbedder(device)
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", subfolder=None).to(device)
        self.vae.eval().requires_grad_(False)

        self.transformer = FluxTransformer(
            in_channels=64, 
            hidden_size=1024,
            depth_double=4,  
            depth_single=8,  
            num_heads=16
        ).to(device)
        
        self.scheduler = FlowMatchScheduler()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        latents = (latents - 0.5) * 2.0
        
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    @torch.no_grad()
    def generate(self, prompt: str, steps: int = 20, guidance_scale: float = 3.5, transformer_override=None):
        model = transformer_override if transformer_override is not None else self.transformer
        model.eval()
        
        txt_pooled, txt_t5 = self.text_encoder([prompt])
        
        num_patches = (self.img_size // 2) ** 2 
        x = torch.randn(1, num_patches, 64).to(self.device)
        
        guidance = torch.tensor([guidance_scale], device=self.device).float()
        
        self.scheduler.set_timesteps(steps, device=self.device)
        
        for i, t in enumerate(self.scheduler.timesteps):
            t_tensor = t.unsqueeze(0).to(self.device) # [1]
            
            v_pred = model(x, t_tensor, guidance, txt_pooled, txt_t5)
            
            if i < len(self.scheduler.timesteps) - 1:
                dt = self.scheduler.timesteps[i+1] - t
            else:
                dt = 0
            
            x = self.scheduler.step(v_pred, t, x, dt)
            
        h = w = int(math.sqrt(x.shape[1]))
        latents = x.view(1, h, w, 16).permute(0, 3, 1, 2)
        latents = F.pixel_shuffle(latents, upscale_factor=2)
        
        return self.decode_latents(latents)