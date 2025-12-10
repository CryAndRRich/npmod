import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

# ==========================================
# PART 1: EMBEDDINGS & HELPERS
# ==========================================

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
    def __init__(self, patch_size: int = 2, in_chans: int = 4, embed_dim: int = 1152):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, Dim, H/p, W/p]
        x = self.proj(x)
        # Flatten: [B, Dim, N] -> [B, N, Dim]
        x = x.flatten(2).transpose(1, 2)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Tạo 2D Sinusoidal Positional Embedding (Style DiT chuẩn)
    Hỗ trợ interpolate cho multi-resolution.
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size
        
    grid_h = int(grid_h)
    grid_w = int(grid_w)

    # Tạo lưới tọa độ
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    
    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()
    
    # Tính embedding cho từng trục
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_y) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_x)

    pos_embed = torch.cat([emb_h, emb_w], dim=1) # [N, Dim]
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

# ==========================================
# PART 2: TRANSFORMER BLOCKS
# ==========================================

class HunyuanDiTBlock(nn.Module):
    """
    Standard DiT Block:
    1. Self-Attention (xử lý ảnh)
    2. Cross-Attention (nhìn vào Text) - Đây là điểm khác biệt với DiT gốc (chỉ dùng AdaLN)
    3. Feed Forward
    4. AdaLN cho Timestep conditioning
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        
        # 1. Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 2. Cross-Attention (Text Condition)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 3. Feed Forward
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # 4. AdaLN (Timestep Scale/Shift)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: Image Tokens [B, N, D]
        # c: Text Context [B, Seq, D] (CLIP + T5 concatenated)
        # t_emb: Timestep Embedding [B, D]
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb)[:, None, :].chunk(6, dim=2)
        
        # Block 1: Self-Attention (Modulated by Time)
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn1(x_norm, x_norm, x_norm)
        x = x + gate_msa * attn_out
        
        # Block 2: Cross-Attention (Conditioned on Text)
        # HunyuanDiT dùng Cross-Attention chuẩn để inject text
        x_norm = self.norm2(x)
        attn_out, _ = self.attn2(x_norm, c, c) # Q=Img, K=Text, V=Text
        x = x + attn_out # Residual chuẩn (thường ko gate hoặc gate cố định)
        
        # Block 3: MLP (Modulated by Time)
        x_norm = self.norm3(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)
        
        return x

# ==========================================
# PART 3: MAIN MODEL & ENCODERS
# ==========================================

class HunyuanTextEmbedder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        
        # 1. CLIP (Bilingual - thường là ViT-L/14)
        self.clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        # 2. T5 (Multilingual - mT5)
        # Hunyuan dùng mT5-XL (rất nặng), ở đây dùng T5-small để demo cấu trúc
        self.t5_tok = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
        self.t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small").to(device)
        
        self.clip_model.eval().requires_grad_(False)
        self.t5_model.eval().requires_grad_(False)
        
        # Projection layer để khớp dimension của CLIP (768) và T5 (512) về hidden_size của DiT
        # Giả sử hidden_size của DiT là 1152
        self.projection = nn.Linear(768 + 512, 1152).to(device)

    def forward(self, prompts: List[str]) -> torch.Tensor:
        # CLIP
        clip_in = self.clip_tok(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clip_out = self.clip_model(clip_in.input_ids).last_hidden_state # [B, 77, 768]
            
        # T5
        t5_in = self.t5_tok(prompts, padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            t5_out = self.t5_model(t5_in.input_ids).last_hidden_state # [B, 128, 512]
            
        # Concatenate & Project
        # Cần padding hoặc truncate để ghép 2 sequence length khác nhau?
        # Hunyuan thường concat theo chiều feature nếu dùng projection, hoặc concat theo chiều sequence.
        # Cách phổ biến nhất (như SDXL): Concat theo chiều sequence nhưng cần project về cùng dim.
        # Ở đây ta dùng cách đơn giản: Pad feature T5 và CLIP về cùng dim rồi concat sequence.
        
        # Cách "From Scratch" đơn giản hóa:
        # Pad CLIP (77 tokens) features lên 1152
        # Pad T5 (128 tokens) features lên 1152
        # Concat -> [B, 77+128, 1152]
        
        clip_proj = F.pad(clip_out, (0, 1152 - 768))
        t5_proj = F.pad(t5_out, (0, 1152 - 512))
        
        context = torch.cat([clip_proj, t5_proj], dim=1) # [B, 205, 1152]
        return context

class DDPMScheduler:
    """
    HunyuanDiT V1.2 dùng Noise Prediction (epsilon)
    """
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012, device="cuda"):
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_train_timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (n,), device=self.device)

    def noise_images(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def step(self, model_output, t, sample):
        # DDIM/DDPM Step đơn giản hóa
        alpha_t = self.alpha_hat[t]
        alpha_prev = self.alpha_hat[t-1] if t > 0 else torch.tensor(1.0).to(self.device)
        
        # Predict x0
        pred_x0 = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        
        # Direction to xt
        dir_xt = torch.sqrt(1 - alpha_prev) * model_output
        
        prev_sample = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
        return prev_sample

class HunyuanDiT(nn.Module):
    def __init__(self, 
                 input_size: int = 64, 
                 patch_size: int = 2, 
                 in_channels: int = 4, 
                 hidden_size: int = 1152, 
                 depth: int = 28, 
                 num_heads: int = 16,
                 device="cuda"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # 1. Encoders
        self.text_encoder = HunyuanTextEmbedder(device)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", subfolder=None).to(device)
        self.vae.eval().requires_grad_(False)
        
        # 2. DiT Backbone
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Positional Embedding (Fixed Sinusoidal 2D)
        # Tính toán trước pos embed cho resolution mặc định
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

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.13025

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.13025 
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    
    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        c = 4 # VAE channels
        p = self.patch_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> Patchify -> [B, N, D]
        x_in = self.x_embedder(x)
        
        # Add Pos Embed (Interpolate nếu size đổi, ở đây demo size cố định)
        if x_in.shape[1] == self.pos_embed.shape[1]:
            x_in = x_in + self.pos_embed
        
        t_emb = self.t_embedder(t)
        
        for block in self.blocks:
            x_in = block(x_in, context, t_emb)
            
        x_out = self.final_layer(x_in)
        
        # Unpatchify: [B, N, P*P*C] -> [B, C, H, W]
        h = w = int(math.sqrt(x_out.shape[1]))
        x_out = self.unpatchify(x_out, h, w)
        
        return x_out
    
    @torch.no_grad()
    def generate(self, prompt: str, steps: int = 50, cfg_scale: float = 7.5, transformer_override=None):
        # Override logic for EMA
        # Lưu ý: "transformer_override" ở đây thực chất là toàn bộ module HunyuanDiT (trừ text encoder/vae)
        # Trong Trainer ta sẽ pass self.model.trainable_module (chính là self)
        
        model = transformer_override if transformer_override is not None else self
        model.eval()
        
        # 1. Text
        cond = self.text_encoder([prompt])
        uncond = self.text_encoder([""])
        context = torch.cat([uncond, cond])
        
        # 2. Init Latents
        latents = torch.randn((1, 4, self.img_size, self.img_size)).to(self.device)
        timesteps = torch.flip(torch.arange(0, self.scheduler.num_train_timesteps, 
                                            self.scheduler.num_train_timesteps // steps), [0]).to(self.device)
        
        # 3. Denoise Loop (DDIM-like)
        for i, t in enumerate(timesteps):
            # Batch input
            latent_input = torch.cat([latents] * 2)
            t_input = torch.cat([t.unsqueeze(0)] * 2)
            
            # Forward
            # Lưu ý: model(x) gọi vào forward -> x_embedder -> blocks
            noise_pred = model(latent_input, t_input, context)
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents = self.scheduler.step(noise_pred, t, latents)
            
        return self.decode_latents(latents)