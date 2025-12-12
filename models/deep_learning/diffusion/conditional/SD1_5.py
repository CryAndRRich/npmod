from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL


class TimeEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Time Embedding Module using sinusoidal embeddings
        
        Parrameters:
            dim: Dimension of the time embeddings
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal time embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_context: int = 768, 
                 n_head: int = 8) -> None:
        """
        Cross-Attention Module
        
        Parameters:
            d_model: Dimension of the model
            d_context: Dimension of the context (for keys and values)
            n_head: Number of attention heads
        """
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = self.d_head ** -0.5
        
        self.to_q = nn.Linear(in_features=d_model, 
                              out_features=d_model, 
                              bias=False)
        self.to_k = nn.Linear(in_features=d_context, 
                              out_features=d_model, 
                              bias=False)
        self.to_v = nn.Linear(in_features=d_context, 
                              out_features=d_model, 
                              bias=False)
        self.to_out = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for cross-attention
        
        Parameters:
            x: Input tensor
            context: Context tensor for keys and values (if None, use x)
        """
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
    def __init__(self, 
                 channels: int, 
                 n_heads: int, 
                 d_context: int = 768) -> None:
        """
        Spatial Transformer Module
        
        Parameters:
            channels: Number of input channels
            n_heads: Number of attention heads
            d_context: Dimension of the context for cross-attention
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, 
                                  num_channels=channels, 
                                  eps=1e-6)
        self.attn1 = CrossAttention(d_model=channels, 
                                    d_context=channels, 
                                    n_head=n_heads) 
        
        self.norm2 = nn.GroupNorm(num_groups=32, 
                                  num_channels=channels, 
                                  eps=1e-6)
        self.attn2 = CrossAttention(d_model=channels, 
                                    d_context=d_context, 
                                    n_head=n_heads) 
        
        self.norm3 = nn.GroupNorm(num_groups=32, 
                                  num_channels=channels, 
                                  eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels * 4),
            nn.GELU(),
            nn.Linear(in_features=channels * 4, out_features=channels)
        )
        self.proj_in = nn.Conv2d(in_channels=channels, 
                                 out_channels=channels, 
                                 kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels=channels, 
                                  out_channels=channels, 
                                  kernel_size=1)

    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_emb_dim: int) -> None:
        """
        Residual Block with time embedding
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_emb_dim: Dimension of the time embeddings
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               padding=1)
        self.time_proj = nn.Linear(in_features=time_emb_dim, 
                                   out_features=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               padding=1)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=1)

    def forward(self, 
                x: torch.Tensor, 
                t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class UNetConditional(nn.Module):
    def __init__(self, 
                 c_in: int = 4, 
                 c_out: int = 4, 
                 time_dim: int = 256, 
                 context_dim: int = 768) -> None:
        """
        Conditional UNet architecture for diffusion models
        
        Parameters:
            c_in: Number of input channels
            c_out: Number of output channels
            time_dim: Dimension of the time embeddings
            context_dim: Dimension of the context for cross-attention
        """
        super().__init__()
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            TimeEmbeddings(dim=time_dim),
            nn.Linear(in_features=time_dim, out_features=time_dim * 4),
            nn.SiLU(),
            nn.Linear(in_features=time_dim * 4, out_features=time_dim * 4),
        )
        
        self.conv_in = nn.Conv2d(in_channels=c_in, 
                                 out_channels=64, 
                                 kernel_size=3, 
                                 padding=1)
        
        self.down1 = ResBlock(in_channels=64, 
                              out_channels=128, 
                              time_emb_dim=time_dim * 4)
        self.trans1 = SpatialTransformer(channels=128, 
                                         n_heads=4, 
                                         d_context=context_dim)
        self.pool1 = nn.Conv2d(in_channels=128, 
                               out_channels=128, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1)
        
        self.down2 = ResBlock(in_channels=128, 
                              out_channels=256, 
                              time_emb_dim=time_dim * 4)
        self.trans2 = SpatialTransformer(channels=256, 
                                         n_heads=8, 
                                         d_context=context_dim)
        self.pool2 = nn.Conv2d(in_channels=256, 
                               out_channels=256, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1)
        
        self.bot1 = ResBlock(in_channels=256, 
                             out_channels=512, 
                             time_emb_dim=time_dim * 4)
        self.bot_trans = SpatialTransformer(channels=512, 
                                            n_heads=8, 
                                            d_context=context_dim)
        self.bot2 = ResBlock(in_channels=512, 
                             out_channels=256, 
                             time_emb_dim=time_dim * 4)
        
        self.up_sample1 = nn.Upsample(scale_factor=2)
        self.up1 = ResBlock(in_channels=256 + 256, 
                            out_channels=128, 
                            time_emb_dim=time_dim * 4) # Concat skip connection
        self.trans_up1 = SpatialTransformer(channels=128, 
                                            n_heads=4, 
                                            d_context=context_dim)
        
        self.up_sample2 = nn.Upsample(scale_factor=2)
        self.up2 = ResBlock(in_channels=128 + 128, 
                            out_channels=64, 
                            time_emb_dim=time_dim * 4)
        self.trans_up2 = SpatialTransformer(channels=64, 
                                            n_heads=4, 
                                            d_context=context_dim)
        
        self.conv_out = nn.Conv2d(in_channels=64, 
                                  out_channels=c_out, 
                                  kernel_size=3, 
                                  padding=1)

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
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
        up2 = self.up2(up2, t_emb)
        up2 = self.trans_up2(up2, context)
        
        return self.conv_out(up2)

class SDScheduler():
    def __init__(self, 
                 noise_steps: int = 1000, 
                 beta_start: float = 0.00085, 
                 beta_end: float = 0.012, 
                 device: str = "cuda") -> None:
        """
        Scheduler for Stable Diffusion noise prediction
        
        Parameters:
            noise_steps: Number of noise steps
            beta_start: Starting value of beta
            beta_end: Ending value of beta
            device: Device to run the scheduler on
        """
        self.device = device
        self.noise_steps = noise_steps
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=0, high=self.noise_steps, size=(n,), device=self.device)

    def noise_images(self, 
                     x: torch.Tensor, 
                     t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def step(self, 
             model_output: torch.Tensor, 
             timestep: int, 
             sample: torch.Tensor) -> torch.Tensor:
        """
        Perform a single step of the scheduler
        
        Parameters:
            model_output: Output from the model (predicted noise)
            timestep: Current timestep
            sample: Current sample (noisy image)
        
        Returns:
            prev_sample: Sample at the previous timestep
        """
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
    def __init__(self, 
                 device: str = "cuda", 
                 max_length: int = 77) -> None:
        """
        CLIP Text Embedder using pre-trained CLIP model
        
        Parameters:
            device: Device to run the model on
            max_length: Maximum length of the tokenized input
        """
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.device = device
        self.max_length = max_length
        self.transformer.eval()

    def forward(self, prompts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(prompts, truncation=True, max_length=self.max_length, 
                               return_length=True, padding="max_length", return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.transformer(input_ids=input_ids)
        return outputs.last_hidden_state

class SD1_5():
    def __init__(self, device: str = "cuda"):
        """
        Stable Diffusion 1.5 Model
        """
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
    def generate(self, 
                 prompt: str, 
                 neg_prompt: str = "", 
                 steps: int = 50, 
                 cfg_scale: float = 7.5, 
                 unet_override: nn.Module = None) -> torch.Tensor:
        """
        Generate an image from a text prompt using Stable Diffusion 1.5
        
        Parameters:
            prompt: The text prompt to generate the image from
            neg_prompt: The negative prompt for classifier-free guidance
            steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            unet_override: Optional UNet model to override the default one
        """
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