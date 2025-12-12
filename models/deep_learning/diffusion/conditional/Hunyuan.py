from typing import List, Tuple
import math

import torch
import torch.nn as nn

from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL


class TimeEmbeddings(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=frequency_embedding_size, 
                      out_features=hidden_size, 
                      bias=True),
            nn.SiLU(),
            nn.Linear(in_features=hidden_size, 
                      out_features=hidden_size, 
                      bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, 
                           dim: int, 
                           max_period: int = 10000) -> None:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class PatchEmbed(nn.Module):
    def __init__(self, 
                 patch_size: int = 2, 
                 in_chans: int = 4, 
                 embed_dim: int = 1152) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def get_2d_sincos_pos_embed(embed_dim: int, 
                            grid_size: Tuple[int, int] | int) -> torch.Tensor:
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size
        
    grid_h = int(grid_h)
    grid_w = int(grid_w)

    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()
    
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_y) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_x)

    pos_embed = torch.cat([emb_h, emb_w], dim=1) 
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, 
                                      pos: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  

    pos = pos.reshape(-1)  
    out = torch.einsum('m,d->md', pos, omega)  

    emb_sin = torch.sin(out) 
    emb_cos = torch.cos(out) 

    emb = torch.cat([emb_sin, emb_cos], dim=1)  
    return emb


class HunyuanDiTBlock(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0) -> None:
        """
        HunyuanDiT Transformer block with Self-Attention, Cross-Attention, MLP, and AdaLN
        
        Parameters:
            hidden_size: Dimension of the hidden features
            num_heads: Number of attention heads
            mlp_ratio: Expansion ratio for the MLP hidden layer
        """
        super().__init__()
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(normalized_shape=hidden_size, 
                                  elementwise_affine=False, 
                                  eps=1e-6)
        self.attn1 = nn.MultiheadAttention(embed_dim=hidden_size, 
                                           num_heads=num_heads, 
                                           batch_first=True)
        
        # Cross-Attention
        self.norm2 = nn.LayerNorm(normalized_shape=hidden_size, 
                                  elementwise_affine=False, 
                                  eps=1e-6)
        self.attn2 = nn.MultiheadAttention(embed_dim=hidden_size, 
                                           num_heads=num_heads, 
                                           batch_first=True)
        
        # Feed Forward
        self.norm3 = nn.LayerNorm(normalized_shape=hidden_size, 
                                  elementwise_affine=False, 
                                  eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(in_features=mlp_hidden_dim, out_features=hidden_size)
        )
        
        # AdaLN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=hidden_size, 
                      out_features=6 * hidden_size, 
                      bias=True)
        )

    def forward(self, 
                x: torch.Tensor, 
                c: torch.Tensor, 
                t_emb: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb)[:, None, :].chunk(6, dim=2)
        
        # Self-Attention
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn1(x_norm, x_norm, x_norm)
        x = x + gate_msa * attn_out
        
        # Cross-Attention
        x_norm = self.norm2(x)
        attn_out, _ = self.attn2(x_norm, c, c) 
        x = x + attn_out 
        
        # MLP
        x_norm = self.norm3(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)
        
        return x


class HunyuanTextEmbedder(nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        
        self.clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        self.t5_tok = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
        self.t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small").to(device)
        
        self.clip_model.eval().requires_grad_(False)
        self.t5_model.eval().requires_grad_(False)

    def forward(self, prompts: List[str]) -> torch.Tensor:
        clip_in = self.clip_tok(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clip_out = self.clip_model(clip_in.input_ids).last_hidden_state 
            
        t5_in = self.t5_tok(prompts, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            t5_out = self.t5_model(t5_in.input_ids).last_hidden_state 
            
        clip_out = clip_out[:, :, :512]
        
        context = torch.cat([clip_out, t5_out], dim=1) 
        return context

class DDPMScheduler():
    def __init__(self, 
                 num_train_timesteps: int = 1000, 
                 beta_start: float = 0.00085, 
                 beta_end: float = 0.012, 
                 device: str = "cuda") -> None:
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_train_timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (n,), device=self.device)

    def noise_images(self, 
                     x: torch.Tensor, 
                     t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def step(self, 
             model_output: torch.Tensor, 
             t: torch.Tensor, 
             sample: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha_hat[t]
        alpha_prev = self.alpha_hat[t-1] if t > 0 else torch.tensor(1.0).to(self.device)
        pred_x0 = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        dir_xt = torch.sqrt(1 - alpha_prev) * model_output
        prev_sample = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
        return prev_sample

class HunyuanDiT(nn.Module):
    def __init__(self, 
                 input_size: int = 64, 
                 patch_size: int = 2, 
                 in_channels: int = 4, 
                 hidden_size: int = 512, 
                 depth: int = 8,       
                 num_heads: int = 8,  
                 device: str = "cuda") -> None:
        """
        HunyuanDiT: A Diffusion Transformer for Image Generation
        
        Parameters:
            input_size: Size of the input images (assumed square)
            patch_size: Size of each image patch
            in_channels: Number of input channels (e.g., 4 for latent space)
            hidden_size: Dimension of the hidden features
            depth: Number of transformer blocks
            num_heads: Number of attention heads in each block
            device: Device to run the model on ("cuda" or "cpu")
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        self.text_encoder = HunyuanTextEmbedder(device)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", subfolder=None).to(device)
        self.vae.eval().requires_grad_(False)
        
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimeEmbeddings(hidden_size)
        
        h = w = input_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(hidden_size, (h, w))
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=False)
        
        self.blocks = nn.ModuleList([
            HunyuanDiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)
        )
        
        self.scheduler = DDPMScheduler(device=device)
        
        self.apply(self._init_weights)
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

        self.to(device)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
            if m.weight is not None: 
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.13025

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.13025 
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    
    def unpatchify(self, 
                   x: torch.Tensor, 
                   h: int, 
                   w: int) -> torch.Tensor:
        c = 4 
        p = self.patch_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        x_in = self.x_embedder(x)
        
        if x_in.shape[1] == self.pos_embed.shape[1]:
            x_in = x_in + self.pos_embed
        
        t_emb = self.t_embedder(t)
        
        for block in self.blocks:
            x_in = block(x_in, context, t_emb)
            
        x_out = self.final_layer(x_in)
        
        h = w = int(math.sqrt(x_out.shape[1]))
        x_out = self.unpatchify(x_out, h, w)
        
        return x_out
    
    @torch.no_grad()
    def generate(self, 
                 prompt: str, 
                 steps: int = 50, 
                 cfg_scale: float = 7.5, 
                 transformer_override = None) -> torch.Tensor:
        model = transformer_override if transformer_override is not None else self
        model.eval()
        
        cond = self.text_encoder([prompt])
        uncond = self.text_encoder([""])
        context = torch.cat([uncond, cond])
        
        latents = torch.randn((1, 4, self.img_size, self.img_size)).to(self.device)
        timesteps = torch.flip(torch.arange(0, self.scheduler.num_train_timesteps, 
                                            self.scheduler.num_train_timesteps // steps), [0]).to(self.device)
        
        for i, t in enumerate(timesteps):
            latent_input = torch.cat([latents] * 2)
            t_input = torch.cat([t.unsqueeze(0)] * 2)
            
            noise_pred = model(latent_input, t_input, context)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents)
            
        return self.decode_latents(latents)