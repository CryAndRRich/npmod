from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from diffusers import AutoencoderKL


class TimeImageEmbeddings(nn.Module):
    def __init__(self, 
                 time_dim: int, 
                 image_emb_dim: int = 1024) -> None:
        """
        Combines time step embeddings with image embeddings
        
        Parameters:
            time_dim: Dimension of the time embeddings
            image_emb_dim: Dimension of the image embeddings
        """
        super().__init__()
        self.time_dim = time_dim
        
        self.time_proj = nn.Linear(time_dim, time_dim)
        self.image_proj = nn.Linear(image_emb_dim, time_dim)
        
        self.act = nn.SiLU()
        self.final = nn.Linear(time_dim, time_dim * 4)

    def forward(self, 
                t: torch.Tensor, 
                image_emb: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.time_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        time_emb = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        time_vec = self.time_proj(time_emb)
        img_vec = self.image_proj(image_emb)
        
        vec = time_vec + img_vec 
        return self.final(self.act(vec))

class KandinskyResBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               padding=1)
        
        self.emb_proj = nn.Linear(in_features=emb_dim, out_features=out_channels)
        
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
                emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class CrossAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_context: int = 768, 
                 n_head: int = 8) -> None:
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
                context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.to_q(x).view(batch_size, seq_len, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = self.to_k(context).view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        v = self.to_v(context).view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.to_out(out)

class KandinskyAttentionBlock(nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_heads: int, 
                 d_context: int = 768) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attn = CrossAttention(d_model=channels, 
                                   d_context=d_context, 
                                   n_head=n_heads)
        self.proj_out = nn.Conv2d(in_channels=channels, 
                                  out_channels=channels, 
                                  kernel_size=1)

    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).view(b, c, -1).permute(0, 2, 1)
        x = self.attn(x, context)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return self.proj_out(x) + x_in


class KandinskyUNet(nn.Module):
    def __init__(self, 
                 c_in: int = 4, 
                 c_out: int = 4, 
                 time_dim: int = 256, 
                 image_emb_dim: int = 1024, 
                 text_emb_dim: int = 768) -> None:
        """
        Kandinsky 2.2 U-Net architecture
        
        Parameters:
            c_in: Number of input channels
            c_out: Number of output channels
            time_dim: Dimension of the time embeddings
            image_emb_dim: Dimension of the image embeddings
            text_emb_dim: Dimension of the text embeddings
        """
        super().__init__()
        self.time_dim = time_dim
        
        self.time_mlp = TimeImageEmbeddings(time_dim, image_emb_dim)
        
        base_ch = 32 
        
        self.conv_in = nn.Conv2d(in_channels=c_in, 
                                 out_channels=base_ch, 
                                 kernel_size=3, 
                                 padding=1)
        
        self.down1 = KandinskyResBlock(in_channels=base_ch, 
                                       out_channels=base_ch * 2, 
                                       emb_dim=time_dim * 4)
        self.attn1 = KandinskyAttentionBlock(channels=base_ch * 2, 
                                             n_heads=4, 
                                             d_context=text_emb_dim)
        self.pool1 = nn.Conv2d(in_channels=base_ch * 2, 
                               out_channels=base_ch * 2, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1)
        
        self.down2 = KandinskyResBlock(in_channels=base_ch * 2, 
                                       out_channels=base_ch * 4, 
                                       emb_dim=time_dim * 4)
        self.attn2 = KandinskyAttentionBlock(channels=base_ch * 4, 
                                             n_heads=8, 
                                             d_context=text_emb_dim)
        self.pool2 = nn.Conv2d(in_channels=base_ch * 4, 
                               out_channels=base_ch * 4, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1)
        
        self.mid1 = KandinskyResBlock(in_channels=base_ch * 4, 
                                      out_channels=base_ch * 8, 
                                      emb_dim=time_dim * 4)
        self.mid_attn = KandinskyAttentionBlock(channels=base_ch * 8, 
                                                n_heads=8, 
                                                d_context=text_emb_dim)
        self.mid2 = KandinskyResBlock(in_channels=base_ch * 8, 
                                      out_channels=base_ch * 4, 
                                      emb_dim=time_dim * 4)
        
        self.up_sample1 = nn.Upsample(scale_factor=2)
        self.up1 = KandinskyResBlock(in_channels=base_ch * 4 + base_ch * 4, 
                                     out_channels=base_ch * 2, 
                                     emb_dim=time_dim * 4)
        self.attn_up1 = KandinskyAttentionBlock(channels=base_ch * 2, 
                                                n_heads=4, 
                                                d_context=text_emb_dim)
        
        self.up_sample2 = nn.Upsample(scale_factor=2)
        self.up2 = KandinskyResBlock(in_channels=base_ch * 2 + base_ch * 2, 
                                     out_channels=base_ch, 
                                     emb_dim=time_dim * 4)
        self.attn_up2 = KandinskyAttentionBlock(channels=base_ch, 
                                                n_heads=4, 
                                                d_context=text_emb_dim)
        
        self.conv_out = nn.Conv2d(in_channels=base_ch, 
                                  out_channels=c_out, 
                                  kernel_size=3, 
                                  padding=1)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                context: torch.Tensor, 
                image_emb: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t, image_emb)
        
        x1 = self.conv_in(x)
        
        x2 = self.down1(x1, emb)
        x2 = self.attn1(x2, context)
        x2_pool = self.pool1(x2)
        
        x3 = self.down2(x2_pool, emb)
        x3 = self.attn2(x3, context)
        x3_pool = self.pool2(x3)
        
        mid = self.mid1(x3_pool, emb)
        mid = self.mid_attn(mid, context)
        mid = self.mid2(mid, emb)
        
        up1 = self.up_sample1(mid)
        up1 = torch.cat((up1, x3), dim=1)
        up1 = self.up1(up1, emb)
        up1 = self.attn_up1(up1, context)
        
        up2 = self.up_sample2(up1)
        up2 = torch.cat((up2, x2), dim=1)
        up2 = self.up2(up2, emb)
        up2 = self.attn_up2(up2, context)
        
        return self.conv_out(up2)


class KandinskyCombinedEmbedder(nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        
        # Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        # Vision Encoder
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        self.text_model.eval().requires_grad_(False)
        self.vision_model.eval().requires_grad_(False)

    def get_text_embeds(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.text_model(tokens.input_ids)
        return outputs.last_hidden_state

    def get_image_embeds(self, images: torch.Tensor) -> torch.Tensor:
        # Resize to 224x224 for CLIP Vision
        images = F.interpolate(images, size=(224, 224), mode="bicubic")
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=images).pooler_output
        return outputs

class DDPMScheduler():
    def __init__(self, 
                 num_train_timesteps: int = 1000, 
                 device: str = "cuda") -> None:
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.beta = torch.linspace(0.00085, 0.012, num_train_timesteps).to(device)
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
        return torch.sqrt(alpha_prev) * pred_x0 + dir_xt


class Kandinsky2_2():
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.img_size = 64
        
        self.embedder = KandinskyCombinedEmbedder(device)
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
        
        self.unet = KandinskyUNet(
            c_in=4, 
            c_out=4, 
            time_dim=256, 
            image_emb_dim=1024,
            text_emb_dim=768
        ).to(device)
        
        self.prior_dummy = nn.Sequential(
            nn.Linear(768, 768), 
            nn.LayerNorm(768)
        ).to(device)
        self.prior_dummy.eval()

        self.scheduler = DDPMScheduler(device=device)
        self.vae.eval().requires_grad_(False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.18215

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.18215 
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    @torch.no_grad()
    def generate(self, 
                 prompt: str, 
                 steps: int = 50, 
                 cfg_scale: float = 4.0, 
                 unet_override = None) -> torch.Tensor:
        unet = unet_override if unet_override is not None else self.unet
        unet.eval()
        
        text_emb_seq = self.embedder.get_text_embeds([prompt])
        text_pooled = text_emb_seq.mean(dim=1)
        image_emb = self.prior_dummy(text_pooled) 
        
        cond_text = self.embedder.get_text_embeds([prompt])
        uncond_text = self.embedder.get_text_embeds([""])
        
        uncond_image_emb = torch.zeros_like(image_emb)
        
        context = torch.cat([uncond_text, cond_text])
        img_embs = torch.cat([uncond_image_emb, image_emb])
        
        latents = torch.randn((1, 4, self.img_size, self.img_size)).to(self.device)
        timesteps = torch.flip(torch.arange(0, self.scheduler.num_train_timesteps, 
                                            self.scheduler.num_train_timesteps // steps), [0]).to(self.device)
        
        for i, t in enumerate(timesteps):
            latent_input = torch.cat([latents] * 2)
            t_input = torch.cat([t.unsqueeze(0)] * 2)
            
            noise_pred = unet(latent_input, t_input, context, img_embs)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents)
            
        return self.decode_latents(latents)