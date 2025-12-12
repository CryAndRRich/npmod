from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL


class TimestepEmbedder(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 frequency_embedding_size: int = 256) -> None:
        """
        Timestep Embedding Module
        
        Parameters:
            hidden_size: The output embedding size
            frequency_embedding_size: The size of the sinusoidal frequency embedding
        """
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
                           max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings

        Parameters:
            t: A 1-D Tensor of timesteps
            dim: The dimension of the embedding
            max_period: The maximum period for the frequencies
        """
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
                 embed_dim: int = 1536) -> None:
        """
        Patch Embedding Module

        Parameters:
            patch_size: The size of each patch
            in_chans: Number of input channels
            embed_dim: Dimension of the embedding
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_chans, 
                              out_channels=embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size, 
                              bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Modulation(nn.Module):
    def __init__(self, 
                 dim: int, 
                 double: bool = True) -> None:
        """
        Modulation Module for Adaptive LayerNorm
        
        Parameters:
            dim: Dimension of the input vector
            double: Whether to output double the parameters (for attention and MLP)
        """
        super().__init__()
        self.is_double = double
        if double:
            self.lin = nn.Linear(in_features=dim, 
                                 out_features=6 * dim, 
                                 bias=True)
        else:
            self.lin = nn.Linear(in_features=dim, 
                                 out_features=3 * dim, 
                                 bias=True)

    def forward(self, vec: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.lin(F.silu(vec))[:, None, :]
        return out.chunk(6 if self.is_double else 3, dim=2)


class RMSNorm(nn.Module):
    def __init__(self, 
                 dim: int, 
                 eps: float = 1e-6) -> None:
        """
        Root Mean Square Layer Normalization
        
        Parameters:
            dim: Dimension of the input tokens
            eps: Small epsilon value for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed
    

class JointAttention(nn.Module):
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8) -> None:
        """
        Joint Attention Module for Image and Text Tokens
        
        Parameters:
            dim: Dimension of the input tokens
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv_x = nn.Linear(in_features=dim, 
                               out_features=dim * 3, 
                               bias=True)
        self.qkv_c = nn.Linear(in_features=dim, 
                               out_features=dim * 3, 
                               bias=True)
        
        self.q_norm_x = RMSNorm(dim=self.head_dim)
        self.k_norm_x = RMSNorm(dim=self.head_dim)
        self.q_norm_c = RMSNorm(dim=self.head_dim)
        self.k_norm_c = RMSNorm(dim=self.head_dim)

        self.proj_x = nn.Linear(in_features=dim, 
                                out_features=dim, 
                                bias=True)
        self.proj_c = nn.Linear(in_features=dim, 
                                out_features=dim, 
                                bias=True)

    def forward(self, 
                x: torch.Tensor, 
                c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_x, C = x.shape
        _, N_c, _ = c.shape

        # Compute QKV separately
        qkv_x = self.qkv_x(x).reshape(B, N_x, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_x, k_x, v_x = qkv_x.unbind(0)

        qkv_c = self.qkv_c(c).reshape(B, N_c, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = qkv_c.unbind(0)

        q_x, k_x = self.q_norm_x(q_x), self.k_norm_x(k_x)
        q_c, k_c = self.q_norm_c(q_c), self.k_norm_c(k_c)

        # Concatenate for Joint Attention
        q = torch.cat([q_x, q_c], dim=2)
        k = torch.cat([k_x, k_c], dim=2)
        v = torch.cat([v_x, v_c], dim=2)

        # Use Scaled Dot Product Attention (Flash Attention compatible)
        x_out = F.scaled_dot_product_attention(q, k, v)

        # Split back to Image and Text
        x_out = x_out.transpose(1, 2).reshape(B, N_x + N_c, C)
        x_out, c_out = x_out.split([N_x, N_c], dim=1)
        
        return self.proj_x(x_out), self.proj_c(c_out)

class MMDiTBlock(nn.Module):
    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0) -> None:
        """
        MM-DiT Block with Joint Attention and Adaptive LayerNorm
        
        Parameters:
            dim: Dimension of the input tokens
            num_heads: Number of attention heads
            mlp_ratio: Ratio for the MLP hidden dimension
        """
        super().__init__()
        self.norm1_x = nn.LayerNorm(normalized_shape=dim, 
                                    elementwise_affine=False, 
                                    eps=1e-6)
        self.norm1_c = nn.LayerNorm(normalized_shape=dim, 
                                    elementwise_affine=False, 
                                    eps=1e-6)
        
        self.attn = JointAttention(dim=dim, num_heads=num_heads)
        
        self.norm2_x = nn.LayerNorm(normalized_shape=dim, 
                                    elementwise_affine=False, 
                                    eps=1e-6)
        self.norm2_c = nn.LayerNorm(normalized_shape=dim, 
                                    elementwise_affine=False, 
                                    eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_x = nn.Sequential(
            nn.Linear(in_features=dim, out_features=mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(in_features=mlp_hidden_dim, out_features=dim)
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(in_features=dim, out_features=mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(in_features=mlp_hidden_dim, out_features=dim)
        )
        
        self.adaLN_modulation = Modulation(dim=dim, double=True)

    def forward(self, 
                x: torch.Tensor, 
                c: torch.Tensor, 
                vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_msa_x, scale_msa_x, gate_msa_x, shift_msa_c, scale_msa_c, gate_msa_c = self.adaLN_modulation(vec)
        
        # Attention
        x_norm = self.norm1_x(x) * (1 + scale_msa_x) + shift_msa_x
        c_norm = self.norm1_c(c) * (1 + scale_msa_c) + shift_msa_c
        
        x_attn, c_attn = self.attn(x_norm, c_norm)
        
        x = x + gate_msa_x * x_attn
        c = c + gate_msa_c * c_attn
        
        # MLP
        x = x + self.mlp_x(self.norm2_x(x)) 
        c = c + self.mlp_c(self.norm2_c(c))
        
        return x, c

class MMDiT(nn.Module):
    def __init__(self, 
                 input_size: int = 64, 
                 patch_size: int = 2, 
                 in_channels: int = 4, # Using 4 to match SDXL VAE
                 hidden_size: int = 1536, 
                 depth: int = 24, 
                 num_heads: int = 24, 
                 text_embed_dim: int = 4096) -> None: 
        """
        Multi-Modal Diffusion Transformer (MMDiT)
        
        Parameters:
            input_size: The height/width of the input image
            patch_size: The size of each patch
            in_channels: Number of input channels
            hidden_size: Dimension of the transformer embeddings
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            text_embed_dim: Dimension of the text embeddings
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        
        self.x_embedder = PatchEmbed(patch_size=patch_size, 
                                     in_chans=in_channels, 
                                     embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)
        self.y_embedder = nn.Linear(in_features=2048, out_features=hidden_size) 
        self.context_embedder = nn.Linear(in_features=text_embed_dim, out_features=hidden_size) 
        
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(data=torch.zeros(1, num_patches, hidden_size))
        
        self.blocks = nn.ModuleList([
            MMDiTBlock(dim=hidden_size, num_heads=num_heads) for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(normalized_shape=hidden_size, 
                                       elementwise_affine=False, 
                                       eps=1e-6)
        self.final_adaLN = Modulation(dim=hidden_size, double=False) 
        self.final_layer = nn.Linear(in_features=hidden_size, 
                                     out_features=patch_size * patch_size * in_channels, 
                                     bias=True)

        self.apply(self._init_weights)
        
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def unpatchify(self, 
                   x: torch.Tensor, 
                   h: int, 
                   w: int) -> torch.Tensor:
        c = self.in_channels
        p = self.patch_size
        
        x = x.reshape(x.shape[0], h // p, w // p, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h, w)
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Embedding
        x = self.x_embedder(x) + self.pos_embed 
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = self.context_embedder(context)
        
        # Global Condition Vector
        vec = t_emb + y_emb
        
        # Transformer Blocks
        for block in self.blocks:
            x, c = block(x, c, vec)
            
        # Final Layer
        shift, scale, gate = self.final_adaLN(vec)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.final_layer(x)
        
        # Unpatchify
        h = w = int(x.shape[1] ** 0.5) * self.patch_size
        x = self.unpatchify(x, h, w)
        
        return x


class FlowMatchScheduler():
    def __init__(self, 
                 num_train_timesteps: int = 1000, 
                 shift: float = 3.0) -> None:
        """
        Flow Matching Scheduler for SD3
        
        Parameters:
            num_train_timesteps: Number of training timesteps
            shift: Shift parameter for the noise schedule
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift 

    def set_timesteps(self, 
                      num_inference_steps: int, 
                      device: str = "cuda") -> None:
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps, device=device)
    
    def step(self, 
             model_output: torch.Tensor, 
             sample: torch.Tensor, 
             dt: float) -> torch.Tensor:
        """
        Perform a single Euler step in the Flow Matching process

        Parameters:
            model_output: The predicted velocity from the model
            sample: The current sample (latent)
            dt: The timestep difference
        
        Returns:
            prev_sample: The updated sample after the Euler step
        """
        prev_sample = sample + model_output * dt
        return prev_sample

class SD3TextEmbedder(nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        
        # CLIP-L/14 (Pooled Output)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        # T5-Small (Sequence Output) - Placeholder for T5-XXL
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
        self.t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small").to(device)
        
        self.clip_model.eval().requires_grad_(False)
        self.t5_model.eval().requires_grad_(False)

    def forward(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # CLIP Processing
        clip_input = self.clip_tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clip_out = self.clip_model(clip_input.input_ids).pooler_output
            
        # Padding CLIP pooled from 768 to 2048 (SD3 standard)
        y = F.pad(clip_out, (0, 2048 - 768)) 

        # T5 Processing
        t5_input = self.t5_tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            t5_out = self.t5_model(t5_input.input_ids).last_hidden_state
        
        # Padding T5-Small (512) to T5-XXL (4096) dimensions
        context = F.pad(t5_out, (0, 4096 - 512))
        
        return y, context


class SD3():
    def __init__(self, device="cuda") -> None:
        self.device = device
        self.img_size = 64 # Latent resolution
        
        self.text_encoder = SD3TextEmbedder(device)
        
        # Use SDXL VAE (4 channels) for compatibility
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", subfolder=None).to(device)
        
        # Initialize MMDiT
        self.transformer = MMDiT(
            input_size=self.img_size,
            patch_size=2,
            in_channels=4, 
            hidden_size=512, 
            depth=8,
            num_heads=8,
            text_embed_dim=4096
        ).to(device)
        
        self.scheduler = FlowMatchScheduler()
        
        # Freeze VAE & Text Encoders
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)
    
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
    def generate(self, 
                 prompt: str, 
                 steps: int = 28, 
                 cfg_scale: float = 7.0, 
                 transformer_override = None) -> torch.Tensor:
        """
        Generate images using Flow Matching.
        Supports transformer_override for EMA inference.
        """
        transformer = transformer_override if transformer_override is not None else self.transformer
        transformer.eval()
        
        # Text Embedding
        y_cond, c_cond = self.text_encoder([prompt])
        y_uncond, c_uncond = self.text_encoder([""]) 
        
        y = torch.cat([y_uncond, y_cond])
        c = torch.cat([c_uncond, c_cond])
        
        # Init Noise (Latent Space)
        latents = torch.randn((1, 4, self.img_size, self.img_size)).to(self.device)
        
        # Flow Matching Loop
        self.scheduler.set_timesteps(steps, device=self.device)
        
        for i, t in enumerate(self.scheduler.timesteps):
            t_batch = torch.cat([t.unsqueeze(0)] * 2)
            latent_batch = torch.cat([latents] * 2)
            
            # Predict Velocity (v)
            noise_pred = transformer(latent_batch, t_batch, y, c)
            
            # Classifier-Free Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            # Calculate dt
            if i < len(self.scheduler.timesteps) - 1:
                dt = self.scheduler.timesteps[i+1] - t
            else:
                dt = 0
                
            # Euler Step
            latents = self.scheduler.step(noise_pred, latents, dt)
            
        return self.decode_latents(latents)