import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

class TimeEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, d_context: int = 768, n_head: int = 8):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = self.d_head ** -0.5
        
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_context, d_model, bias=False)
        self.to_v = nn.Linear(d_context, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if context is None:
            context = x

        batch_size, seq_len, _ = x.shape
        
        q = self.to_q(x).view(batch_size, seq_len, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = self.to_k(context).view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        v = self.to_v(context).view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.to_out(out)

class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, d_context: int = 768):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels, eps=1e-6)
        self.attn1 = CrossAttention(channels, channels, n_heads) 
        
        self.norm2 = nn.GroupNorm(32, channels, eps=1e-6)
        self.attn2 = CrossAttention(channels, d_context, n_heads) 
        
        self.norm3 = nn.GroupNorm(32, channels, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = x.view(b, c, -1).permute(0, 2, 1)
        
        x_norm = self.norm1(x_in).view(b, c, -1).permute(0, 2, 1)
        x = x + self.attn1(x_norm, context=None)
        
        x_norm = self.norm2(x.permute(0, 2, 1).view(b, c, h, w)).view(b, c, -1).permute(0, 2, 1)
        x = x + self.attn2(x_norm, context=context)
        
        x_norm = self.norm3(x.permute(0, 2, 1).view(b, c, h, w)).view(b, c, -1).permute(0, 2, 1)
        x = x + self.ff(x_norm)
        
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return self.proj_out(x) + self.proj_in(x_in)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class UNetConditional(nn.Module):
    def __init__(self, c_in: int=4, c_out: int=4, time_dim: int=256, context_dim: int=768):
        super().__init__()
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            TimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )
        
        self.conv_in = nn.Conv2d(c_in, 64, kernel_size=3, padding=1)
        
        self.down1 = ResBlock(64, 128, time_dim * 4)
        self.trans1 = SpatialTransformer(128, 4, context_dim)
        self.pool1 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        
        self.down2 = ResBlock(128, 256, time_dim * 4)
        self.trans2 = SpatialTransformer(256, 8, context_dim)
        self.pool2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        
        self.bot1 = ResBlock(256, 512, time_dim * 4)
        self.bot_trans = SpatialTransformer(512, 8, context_dim)
        self.bot2 = ResBlock(512, 256, time_dim * 4)
        
        self.up_sample1 = nn.Upsample(scale_factor=2)
        self.up1 = ResBlock(256 + 256, 128, time_dim * 4) # Concat skip connection
        self.trans_up1 = SpatialTransformer(128, 4, context_dim)
        
        self.up_sample2 = nn.Upsample(scale_factor=2)
        self.up2 = ResBlock(128 + 128, 64, time_dim * 4)
        self.trans_up2 = SpatialTransformer(64, 4, context_dim)
        
        self.conv_out = nn.Conv2d(64, c_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        
        x1 = self.conv_in(x) 
        x2 = self.down1(x1, t_emb)
        x2 = self.trans1(x2, context)
        x2_pool = self.pool1(x2)
        
        x3 = self.down2(x2_pool, t_emb)
        x3 = self.trans2(x3, context)
        x3_pool = self.pool2(x3)
        
        bot = self.bot1(x3_pool, t_emb)
        bot = self.bot_trans(bot, context)
        bot = self.bot2(bot, t_emb)
        
        up1 = self.up_sample1(bot)
        up1 = torch.cat((up1, x3), dim=1)
        up1 = self.up1(up1, t_emb)
        up1 = self.trans_up1(up1, context)
        
        up2 = self.up_sample2(up1)
        up2 = torch.cat((up2, x2), dim=1)
        up2 = self.trans_up2(up2, context)
        
        return self.conv_out(up2)

class SDScheduler:
    def __init__(self, noise_steps=1000, beta_start=0.00085, beta_end=0.012, device="cuda"):
        self.device = device
        self.noise_steps = noise_steps
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def step(self, model_output, timestep, sample):
        t = timestep
        prev_t = t - 1
        
        alpha_prod_t = self.alpha_hat[t]
        alpha_prod_t_prev = self.alpha_hat[prev_t] if prev_t >= 0 else torch.tensor(1.0).to(self.device)
        
        beta_prod_t = 1 - alpha_prod_t
        
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
        
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        return prev_sample

class CLIPTextEmbedder(nn.Module):
    def __init__(self, device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.device = device
        self.max_length = max_length
        self.transformer.eval()

    def forward(self, prompts: List[str]):
        batch = self.tokenizer(prompts, truncation=True, max_length=self.max_length, 
                               return_length=True, padding="max_length", return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.transformer(input_ids=input_ids)
        return outputs.last_hidden_state

class SD1_5():
    def __init__(self, device="cuda"):
        self.device = device
        self.img_size = 64
        
        self.text_encoder = CLIPTextEmbedder(device)
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
        self.unet = UNetConditional(c_in=4, c_out=4, time_dim=256, context_dim=768).to(device)
        self.scheduler = SDScheduler(device=device)

        self.vae.eval().requires_grad_(False)
        self.text_encoder.transformer.eval().requires_grad_(False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.18215

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs.cpu().permute(0, 2, 3, 1).numpy()

    @torch.no_grad()
    def generate(self, prompt: str, neg_prompt: str = "", steps: int = 50, cfg_scale: float = 7.5, unet_override=None):
        unet = unet_override if unet_override is not None else self.unet
        unet.eval()
        
        cond_emb = self.text_encoder([prompt])
        uncond_emb = self.text_encoder([neg_prompt])
        context = torch.cat([uncond_emb, cond_emb])
        
        latents = torch.randn((1, 4, self.img_size, self.img_size)).to(self.device)
        timesteps = torch.linspace(self.scheduler.noise_steps - 1, 0, steps).long().to(self.device)
        
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            t_in = torch.cat([t.unsqueeze(0)] * 2)
            
            noise_pred = unet(latent_model_input, t_in, context)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents)
            
        return self.decode_latents(latents)