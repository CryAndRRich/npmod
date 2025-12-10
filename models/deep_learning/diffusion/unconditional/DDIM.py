import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Sinusoidal time embeddings
        
        Parameters:
            dim: Dimension of the time embeddings
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time embeddings

        Parameters:
            time: Representing time steps
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, 
                 channels: int, 
                 size: int) -> None:
        """
        Self-Attention module
        
        Parameters:
            channels: Number of input channels
            size: Spatial size of the input feature map
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=channels, 
                                         num_heads=4, 
                                         batch_first=True)
        self.ln = nn.LayerNorm(normalized_shape=[channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm(normalized_shape=[channels]),
            nn.Linear(in_features=channels, out_features=channels),
            nn.GELU(),
            nn.Linear(in_features=channels, out_features=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 mid_channels: int = None, 
                 residual: bool = False) -> None:
        """
        Double Convolution Block with optional residual connection
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels
            residual: Whether to use residual connection
        """
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=mid_channels, 
                      kernel_size=3, 
                      padding=1, 
                      bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels, 
                      out_channels=out_channels, 
                      kernel_size=3, 
                      padding=1, 
                      bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 emb_dim: int = 256) -> None:
        """
        Downsampling Block with time embedding
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            emb_dim: Dimension of the time embeddings
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, 
                       out_channels=in_channels, 
                       residual=True),
            DoubleConv(in_channels=in_channels, 
                       out_channels=out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor) -> torch.Tensor:
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 emb_dim: int = 256) -> None:
        """
        Upsampling Block with time embedding
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            emb_dim: Dimension of the time embeddings
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, 
                              mode="bilinear", 
                              align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, 
                       out_channels=in_channels, 
                       residual=True),
            DoubleConv(in_channels=in_channels, 
                       out_channels=out_channels, 
                       mid_channels=in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, 
                x: torch.Tensor, 
                skip_x: torch.Tensor, 
                t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class UNet(nn.Module):
    def __init__(self, 
                 c_in: int = 3, 
                 c_out: int = 3, 
                 time_dim: int = 256, 
                 device: str = "cuda") -> None:
        """
        U-Net architecture for Denoising Diffusion Probabilistic Models
        
        Parameters:
            c_in: Number of input channels
            c_out: Number of output channels
            time_dim: Dimension of the time embeddings
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            TimeEmbeddings(dim=time_dim),
            nn.Linear(in_features=time_dim, out_features=time_dim),
            nn.ReLU()
        )
        
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16) 
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)  

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128) 
        self.sa4 = SelfAttention(128, 16) 
        self.up2 = Up(256, 64) 
        self.up3 = Up(128, 64) 
        
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor) -> torch.Tensor:
        t = t.to(self.device)
        t = self.time_mlp(t)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        
        output = self.outc(x)
        return output
        
class DiffusionUtils():
    def __init__(self, 
                 noise_steps: int = 1000, 
                 beta_start: float = 1e-4, 
                 beta_end: float = 0.02, 
                 img_size: int = 64, 
                 device: str = "cuda", 
                 schedule_name: str = "cosine") -> None:
        """
        Diffusion Utilities for Denoising Diffusion Probabilistic Models
        
        Parameters:
            noise_steps: Number of noise steps
            beta_start: Starting value of beta
            beta_end: Ending value of beta
            img_size: Size of the input images
            device: Device to run the computations on
            schedule_name: Type of noise schedule ("linear" or "cosine")
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule_name = schedule_name 

        if self.schedule_name == "linear":
            self.beta = self.prepare_noise_schedule().to(device)
        elif self.schedule_name == "cosine":
            self.beta = self.prepare_cosine_schedule().to(device)
            
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self) -> torch.Tensor: 
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def prepare_cosine_schedule(self, s:float = 0.008) -> torch.Tensor:
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps)
        alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def noise_images(self, 
                     x: torch.Tensor, 
                     t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=0, high=self.noise_steps, size=(n,), device=self.device)

    @torch.no_grad()
    def sample_ddim(self, 
                    model: nn.Module, 
                    n: int, 
                    ddim_timesteps: int = 50, 
                    eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling procedure
        
        Parameters:
            model: The trained diffusion model
            n: Number of samples to generate
            ddim_timesteps: Number of DDIM timesteps
            eta: Controls the scale of added noise
        """
        model.eval()
        inference_indices = torch.linspace(0, self.noise_steps - 1, ddim_timesteps).long()
        inference_indices = torch.flip(inference_indices, [0]).tolist()

        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

        for i, step in enumerate(inference_indices):
            t = (torch.ones(n) * step).long().to(self.device)
            predicted_noise = model(x, t)
            
            alpha_hat_t = self.alpha_hat[t][:, None, None, None]
            
            if i < len(inference_indices) - 1:
                next_step = inference_indices[i+1]
                alpha_hat_t_prev = self.alpha_hat[next_step].view(1, 1, 1, 1)
            else:
                alpha_hat_t_prev = torch.tensor(1.0).to(self.device).view(1, 1, 1, 1)

            pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            sigma_t = eta * torch.sqrt((1 - alpha_hat_t_prev) / (1 - alpha_hat_t) * (1 - alpha_hat_t / alpha_hat_t_prev))
            
            if i == len(inference_indices) - 1:
                sigma_t = 0

            dir_xt = torch.sqrt(1 - alpha_hat_t_prev - sigma_t**2) * predicted_noise
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt + sigma_t * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x