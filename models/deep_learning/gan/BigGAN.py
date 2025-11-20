from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from ..gan import GAN

def update_ema(ema_model: nn.Module, 
               model: nn.Module, 
               decay: float) -> None:
    """
    Update the EMA (Exponential Moving Average) model parameters
    
    Parameters:
        ema_model: The EMA model to update
        model: The current model
        decay: The decay rate for EMA
    """
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if k in msd:
                v.copy_(v * decay + msd[k].detach() * (1 - decay))

def orthogonal_regularization(model: nn.Module, 
                              strength: float = 1e-4) -> torch.Tensor:
    """
    Compute orthogonal regularization loss for the model
    
    Parameters:
        model: The model to regularize
        strength: The strength of the regularization
    """
    reg = 0.0
    for p in model.parameters():
        if p.ndim >= 2:
            w = p.view(p.size(0), -1)
            wt_w = torch.matmul(w, w.t())
            ident = torch.eye(wt_w.size(0), device=wt_w.device)
            reg = reg + ((wt_w - ident).pow(2).sum())
    return strength * reg

class SelfAttention(nn.Module):
    def __init__(self, in_dim: int) -> None:
        """
        Self-Attention layer as used in BigGAN
        
        Parameters:
            in_dim: Number of input channels
        """
        super().__init__()
        self.query_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, 
                                                  out_channels=in_dim // 8, 
                                                  kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, 
                                                out_channels=in_dim // 8, 
                                                kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, 
                                                  out_channels=in_dim, 
                                                  kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, W, H = x.size()
        
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        
        return self.gamma * out + x

class ResBlockUp(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int) -> None:
        """
        Residual block with upsampling
        
        Parameters:
            in_ch: Number of input channels
            out_ch: Number of output channels
        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_ch)
        self.conv1 = nn.Conv2d(in_channels=in_ch, 
                               out_channels=out_ch, 
                               kernel_size=3, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, 
                               out_channels=out_ch, 
                               kernel_size=3, 
                               padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if in_ch != out_ch:
            self.conv_short = nn.Conv2d(in_channels=in_ch, 
                                        out_channels=out_ch, 
                                        kernel_size=1)
        else:
            self.conv_short = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection
        sc = self.upsample(x)
        sc = self.conv_short(sc) if self.conv_short is not None else sc

        out = self.bn1(x)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + sc

class ResBlockDown(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int) -> None:
        """
        Residual block with downsampling
        
        Parameters:
            in_ch: Number of input channels
            out_ch: Number of output channels
        """
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=in_ch, 
                                             out_channels=in_ch, 
                                             kernel_size=3, 
                                             padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=in_ch, 
                                             out_channels=out_ch, 
                                             kernel_size=3, 
                                             padding=1))
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if in_ch != out_ch:
            self.conv_short = spectral_norm(nn.Conv2d(in_channels=in_ch, 
                                                      out_channels=out_ch, 
                                                      kernel_size=1))
        else:
            self.conv_short = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection
        sc = self.downsample(x)
        if self.conv_short:
            sc = self.conv_short(sc)

        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.downsample(out)
        return out + sc

class Generator(nn.Module):
    def __init__(self,
                 z_dim: int = 128,
                 base_ch: int = 32,
                 img_channels: int = 3,
                 resolution: int = 64,
                 attention_res: int = 64) -> None:
        """
        BigGAN Generator

        Parameters:
            z_dim: Dimension of input noise vector
            base_ch: Base channel multiplier (controls model size)
            img_channels: Number of output image channels
            resolution: Output image resolution (assumed square)
            attention_res: Resolution at which to apply self-attention
        """
        super().__init__()
        assert resolution in (32, 64, 128, 256), "Only supported resolutions: 32, 64, 128, 256"
        self.z_dim = z_dim
        self.resolution = resolution
        self.attention_res = attention_res

        if resolution == 128:
            mults = [16, 8, 4, 2, 1]  
        elif resolution == 64:
            mults = [16, 8, 4, 1]
        elif resolution == 256:
            mults = [32, 16, 8, 4, 2, 1]
        else: 
            mults = [8, 4, 2, 1]

        chs = [base_ch * m for m in mults]
        self.init_ch = chs[0]
        
        self.project = spectral_norm(nn.Linear(in_features=z_dim, out_features=4 * 4 * self.init_ch))

        self.layers = nn.ModuleList([])
        in_ch = self.init_ch
        curr_res = 4
        
        for out_ch in chs[1:]:
            self.layers.append(ResBlockUp(in_ch, out_ch))
            in_ch = out_ch
            curr_res = curr_res * 2
            
            if curr_res == self.attention_res:
                self.layers.append(SelfAttention(in_ch))

        self.bn_last = nn.BatchNorm2d(num_features=in_ch)
        self.conv_out = spectral_norm(nn.Conv2d(in_channels=in_ch, 
                                                out_channels=img_channels, 
                                                kernel_size=3, 
                                                padding=1))
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.project(x).view(b, self.init_ch, 4, 4)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.bn_last(x)
        x = F.relu(x)
        x = self.conv_out(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,
                 img_channels: int = 3,
                 base_ch: int = 32,
                 resolution: int = 64,
                 attention_res: int = 64) -> None:
        """
        BigGAN Discriminator

        Parameters:
            img_channels: Number of input image channels
            base_ch: Base channel multiplier (controls model size)
            resolution: Input image resolution (assumed square)
            attention_res: Resolution at which to apply self-attention
        """
        super().__init__()
        assert resolution in (32, 64, 128, 256), "Only supported resolutions: 32, 64, 128, 256"
        self.resolution = resolution
        self.attention_res = attention_res
        
        if resolution == 128:
            mults = [1, 2, 4, 8, 16]
        elif resolution == 64:
            mults = [1, 4, 8, 16]
        elif resolution == 256:
            mults = [1, 2, 4, 8, 16, 32]
        else:
            mults = [1, 2, 4, 8]

        chs = [base_ch * m for m in mults]
        
        self.layers = nn.ModuleList([])
        
        self.layers.append(
            spectral_norm(nn.Conv2d(in_channels=img_channels,
                                    out_channels=chs[0], 
                                    kernel_size=3, 
                                    padding=1))
        )
        
        curr_res = resolution
        for i in range(len(chs) - 1):
            self.layers.append(ResBlockDown(chs[i], chs[i + 1]))
            curr_res = curr_res // 2
            
            if curr_res == self.attention_res:
                self.layers.append(SelfAttention(chs[i + 1]))

        self.act = nn.ReLU()
        self.fc = spectral_norm(nn.Linear(in_features=chs[-1], out_features=1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h = img
        for layer in self.layers:
            h = layer(h)
            
        h = self.act(h)
        h = h.view(h.size(0), h.size(1), -1).sum(dim=2)
        out = self.fc(h)
        return out.view(-1)

class BigGAN(GAN):
    def __init__(self,
                 latent_dim: int = 128,
                 img_shape: Tuple[int, int, int] = (3, 128, 128),
                 learn_rate: float = 2e-4,
                 number_of_epochs: int = 100,
                 base_ch: int = 32,
                 truncation: float = 1.0,
                 ortho_reg: bool = False,
                 ortho_strength: float = 1e-5,
                 ema_decay: float = 0.999,
                 ema_every: int = 10,
                 n_critic: int = 1) -> None:
        """
        Unconditional BigGAN

        Parameters:
            latent_dim: Dimension of the input noise vector
            img_shape: Shape of the generated images (C, H, W)
            learn_rate: Learning rate for optimizers
            number_of_epochs: Number of training epochs
            base_ch: Base channel multiplier for model size
            truncation: Truncation value for input noise
            ortho_reg: Whether to apply orthogonal regularization
            ortho_strength: Strength of orthogonal regularization
            ema_decay: Decay rate for EMA of generator
            ema_every: Frequency (in steps) to update EMA
            n_critic: Number of discriminator updates per generator update
        """
        super().__init__(
            latent_dim=latent_dim,
            img_shape=img_shape,
            learn_rate=learn_rate,
            number_of_epochs=number_of_epochs
        )
        self.base_ch = base_ch
        self.truncation = truncation
        self.ortho_reg = ortho_reg
        self.ortho_strength = ortho_strength
        self.ema_decay = ema_decay
        self.ema_every = ema_every
        self.n_critic = n_critic

    @staticmethod
    def hinge_d_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        real_loss = torch.mean(F.relu(1.0 - real_scores))
        fake_loss = torch.mean(F.relu(1.0 + fake_scores))
        return real_loss + fake_loss

    @staticmethod
    def hinge_g_loss(fake_scores: torch.Tensor) -> torch.Tensor:
        return -torch.mean(fake_scores)

    def init_network(self) -> None:
        C, H, _ = self.img_shape
        resolution = H
        self.generator = Generator(self.latent_dim, self.base_ch, C, resolution)
        self.discriminator = Discriminator(C, self.base_ch, resolution)
        # EMA generator
        self.generator_ema = Generator(self.latent_dim, self.base_ch, C, resolution)

        self.generator.apply(lambda m: self.init_weights(m, type="orthogonal"))
        self.discriminator.apply(lambda m: self.init_weights(m, type="orthogonal"))
        self.generator_ema.load_state_dict(self.generator.state_dict())

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.0, 0.9))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.0, 0.9))

    def fit(self,
            dataloader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        self.init_network()
        self.generator.train()
        self.discriminator.train()

        step = 0
        for epoch in range(self.number_of_epochs):
            epoch_loss_G, epoch_loss_D = 0.0, 0.0
            for real_images, _ in dataloader:
                batch_size = real_images.size(0)

                # Train Discriminator
                for _ in range(self.n_critic):
                    z = torch.randn(batch_size, self.latent_dim)
                    fake_images = self.generator(z).detach()

                    real_scores = self.discriminator(real_images)
                    fake_scores = self.discriminator(fake_images)

                    d_loss = self.hinge_d_loss(real_scores, fake_scores)

                    if self.ortho_reg and self.ortho_strength > 0.0:
                        d_loss = d_loss + orthogonal_regularization(self.discriminator, self.ortho_strength)

                    self.optimizer_D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()

                # Train Generator
                z = torch.randn(batch_size, self.latent_dim)
                fake_images = self.generator(z)
                fake_scores = self.discriminator(fake_images)
                g_loss = self.hinge_g_loss(fake_scores)

                if self.ortho_reg and self.ortho_strength > 0.0:
                    g_loss = g_loss + orthogonal_regularization(self.generator, self.ortho_strength)

                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                # EMA update
                step += 1
                if step % self.ema_every == 0:
                    update_ema(self.generator_ema, self.generator, self.ema_decay)

                epoch_loss_G += g_loss.item()
                epoch_loss_D += d_loss.item()

            epoch_loss_G /= len(dataloader)
            epoch_loss_D /= len(dataloader)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | "
                      f"Loss_D: {epoch_loss_D:.4f} | Loss_G: {epoch_loss_G:.4f}")

    def generate(self, z: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        model = self.generator_ema if (use_ema and self.generator_ema is not None) else self.generator
        model.eval()
        with torch.no_grad():
            z_in = z
            if self.truncation < 1.0:
                z_in = torch.clamp(z_in, -self.truncation, self.truncation)
            imgs = model(z_in)
            imgs = (imgs.clamp(-1.0, 1.0) + 1.0) / 2.0
        return imgs

    def __str__(self) -> str:
        return "BigGAN"