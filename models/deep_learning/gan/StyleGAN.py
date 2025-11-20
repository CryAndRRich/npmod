import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from ..gan import GAN

class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class MappingNetwork(nn.Module):
    def __init__(self, 
                 latent_dim: int = 512, 
                 dlatent_dim: int = 512, 
                 num_layers: int = 8) -> None:
        """
        Mapping network from z to w

        Parameters:
            latent_dim: Dimension of input noise vector z
            dlatent_dim: Dimension of intermediate latent vector w
            num_layers: Number of layers in mapping network
        """
        super().__init__()
        layers = [PixelNorm()]
        in_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=in_dim, out_features=dlatent_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_dim = dlatent_dim
        self.mapping = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), -1)
        w = self.mapping(z)
        return w

class AdaIN(nn.Module):
    def __init__(self, 
                 channels: int, 
                 dlatent_dim: int = 512) -> None:
        """
        Adaptive Instance Normalization (AdaIN)

        Parameters:
            channels: Number of feature map channels
            dlatent_dim: Dimension of style vector w
        """
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features=channels)
        self.style_scale = nn.Linear(in_features=dlatent_dim, out_features=channels)
        self.style_shift = nn.Linear(in_features=dlatent_dim, out_features=channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        style_scale = self.style_scale(w).view(B, C, 1, 1)
        style_shift = self.style_shift(w).view(B, C, 1, 1)
        x_norm = self.norm(x)
        return style_scale * x_norm + style_shift

class StyledConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 dlatent_dim: int = 512, 
                 upsample: bool = False) -> None:
        """
        Styled convolutional block with optional upsampling

        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dlatent_dim: Dimension of style vector w
            upsample: Whether to upsample by 2x
        """
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.upsampler = nn.Upsample(scale_factor=2, 
                                         mode="bilinear", 
                                         align_corners=False)
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
        self.adain = AdaIN(channels=out_channels, dlatent_dim=dlatent_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.noise_strength = nn.Parameter(data=torch.zeros(1, out_channels, 1, 1))
        self.bias = nn.Parameter(data=torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.upsampler(x)
        x = self.conv(x)
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3))
        x = x + self.noise_strength * noise
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.adain(x, w)
        x = self.lrelu(x)
        return x

class ToRGB(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class Generator(nn.Module):
    def __init__(self,
                 latent_dim: int = 512,
                 img_shape: Tuple[int, int, int] = (3, 64, 64),
                 dlatent_dim: int = 512,
                 fmap_base: int = 512,
                 mixing_prob: float = 0.9) -> None:
        """
        StyleGAN Generator 

        Parameters:
            latent_dim: Dimension of input noise vector 
            img_shape: Shape of output image
            dlatent_dim: Dimension of intermediate latent vector w
            fmap_base: Base number of feature maps
            mixing_prob: Probability of style mixing
        """
        super().__init__()
        C, H, W = img_shape
        assert H == W and (H & (H - 1)) == 0 and H >= 8, "H must be power of two and H >= 8"
        self.latent_dim = latent_dim
        self.dlatent_dim = dlatent_dim
        self.C = C
        self.H = H
        self.n_layers = int(math.log2(H)) - 2

        self.mapping = MappingNetwork(latent_dim, dlatent_dim, num_layers=8)
        self.const_input = nn.Parameter(data=torch.randn(1, fmap_base, 4, 4))
        self.initial = StyledConvBlock(fmap_base, fmap_base, dlatent_dim, upsample=False)
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_c = fmap_base
        for i in range(self.n_layers):
            out_c = max(fmap_base // (2 ** (i + 1)), 32)
            block = StyledConvBlock(in_c, out_c, dlatent_dim, upsample=True)
            self.blocks.append(block)
            self.to_rgbs.append(ToRGB(out_c, C))
            in_c = out_c

        if len(self.to_rgbs) == 0:
            self.to_rgbs.append(ToRGB(fmap_base, C))

        self.mixing_prob = mixing_prob

    def make_ws(self, 
                z: torch.Tensor, 
                inject_index: Optional[int] = None) -> torch.Tensor:
        """
        Generate style vectors w from input noise z, with optional style mixing
        """
        w = self.mapping(z)  
        n_total = 1 + self.n_layers  
        if inject_index is None and torch.rand(1).item() < self.mixing_prob:
            z2 = torch.randn_like(z)
            w2 = self.mapping(z2)
            cutoff = torch.randint(1, n_total, (1,)).item()
            ws = []
            for i in range(n_total):
                ws.append(w if i < cutoff else w2)
            ws = torch.stack(ws, dim=1) 
        else:
            ws = w.unsqueeze(1).repeat(1, n_total, 1)
        return ws
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of StyleGAN generator

        Parameters:
            z: Input noise tensor 
        """
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        B = z.size(0)
        ws = self.make_ws(z) 
        x = self.const_input.repeat(B, 1, 1, 1)  
        x = self.initial(x, ws[:, 0])
        rgb = None
        for i, block in enumerate(self.blocks):
            x = block(x, ws[:, 1 + i])
            rgb_new = self.to_rgbs[i](x)
            if rgb is None:
                rgb = rgb_new
            else:
                rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False) + rgb_new
        if rgb is None:
            rgb = self.to_rgbs[0](x)
        img = torch.tanh(rgb)
        return img

def minibatch_stddev(x: torch.Tensor, 
                     group_size: int = 4, 
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Minibatch Standard Deviation Layer
    
    Parameters:
        x: Input tensor of shape
        group_size: Size of groups for computing stddev
        eps: Small value to avoid division by zero
    
    Returns:
        Tensor with appended minibatch stddev feature map
    """
    B, C, H, W = x.shape
    G = min(group_size, B)
    if B % G != 0:
        G = B
    y = x.view(G, -1, C, H, W) 
    y = y - y.mean(dim=0, keepdim=True)
    var = (y ** 2).mean(dim=0)
    std = torch.sqrt(var + eps)
    mean_std = std.mean(dim=(1, 2, 3), keepdim=True) 
    mean_std = mean_std.repeat(G, 1, H, W).view(B, 1, H, W)
    return torch.cat([x, mean_std], dim=1)

class Discriminator(nn.Module):
    def __init__(self, 
                 img_shape: Tuple[int, int, int], 
                 fmap_base: int = 512) -> None:
        """
        Discriminator for StyleGAN

        Parameters:
            img_shape: Shape of input image 
            fmap_base: Base number of feature maps
        """
        super().__init__()
        C, H, W = img_shape
        assert H == W and (H & (H - 1)) == 0 and H >= 8
        self.C = C
        self.H = H
        layers = []
        in_c = C

        n_blocks = int(math.log2(H)) - 2
        for i in range(n_blocks):
            out_c = max(fmap_base // (2 ** (i)), 32)
            layers.append(nn.Conv2d(in_channels=in_c, 
                                    out_channels=out_c, 
                                    kernel_size=4, 
                                    stride=2,
                                    padding=1))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_c = out_c

        self.conv_blocks = nn.Sequential(*layers)
        final_spatial = 4 
        self.final_conv = nn.Conv2d(in_channels=in_c + 1, 
                                    out_channels=in_c, 
                                    kernel_size=3, 
                                    padding=1)
        self.final_lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc1 = nn.Linear(in_features=in_c * final_spatial * final_spatial, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator

        Parameters:
            img: Input image tensor

        Returns:
            Validity score (probability)
        """
        x = self.conv_blocks(img)
        x = minibatch_stddev(x)
        x = self.final_conv(x)
        x = self.final_lrelu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.final_lrelu(x)
        x = self.fc2(x)
        return x

class StyleGAN(GAN):
    def __init__(self,
                 latent_dim: int = 512,
                 img_shape: Tuple[int, int, int] = (3, 64, 64),
                 dlatent_dim: int = 512,
                 learn_rate: float = 2e-4,
                 number_of_epochs: int = 100,
                 r1_gamma: float = 10.0,
                 n_critic: int = 1,
                 fmap_base: int = 512) -> None:
        """
        StyleGAN model
        
        Parameters:
            latent_dim: Dimension of input noise vector
            img_shape: Shape of input image
            dlatent_dim: Dimension of intermediate latent vector w
            learn_rate: Learning rate for optimizers
            number_of_epochs: Number of training epochs
            r1_gamma: Weight for R1 regularization
            n_critic: Number of discriminator updates per generator update
            fmap_base: Base number of feature maps
        """
        super().__init__(
            latent_dim=latent_dim, 
            img_shape=img_shape, 
            learn_rate=learn_rate, 
            number_of_epochs=number_of_epochs
        )
        self.dlatent_dim = dlatent_dim
        self.r1_gamma = r1_gamma
        self.n_critic = n_critic
        self.fmap_base = fmap_base

    def init_network(self) -> None:
        self.generator = Generator(self.latent_dim, self.img_shape, self.dlatent_dim, self.fmap_base)
        self.discriminator = Discriminator(self.img_shape, self.fmap_base)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.0, 0.99))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.0, 0.99))

    @staticmethod
    def d_logistic_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
        return loss

    @staticmethod
    def g_nonsat_loss(fake_logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(-fake_logits).mean()

    @staticmethod
    def r1_regularization(real_generated_images: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        grads = autograd.grad(outputs=real_logits.sum(), inputs=real_generated_images, create_graph=True)[0]
        return grads.pow(2).reshape(grads.size(0), -1).sum(1).mean()

    def fit(self, 
            dataloader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        self.init_network()
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.number_of_epochs):
            epoch_loss_G, epoch_loss_D = 0.0, 0.0

            for real_images, _ in dataloader:
                batch_size = real_images.size(0)
                
                # Train Discriminator
                for _ in range(self.n_critic):
                    z = torch.randn(batch_size, self.latent_dim)
                    fake_images = self.generator(z).detach()
                    real_logits = self.discriminator(real_images)
                    fake_logits = self.discriminator(fake_images)

                    d_loss = self.d_logistic_loss(real_logits, fake_logits)

                    self.optimizer_D.zero_grad()
                    d_loss.backward()
                    if self.r1_gamma is not None and self.r1_gamma > 0.0:
                        real_images.requires_grad_(True)
                        real_logits_for_r1 = self.discriminator(real_images)
                        r1 = self.r1_regularization(real_images, real_logits_for_r1)
                        (0.5 * self.r1_gamma * r1).backward()
                        real_images.requires_grad_(False)
                    self.optimizer_D.step()

                # Train Generator
                z = torch.randn(batch_size, self.latent_dim)
                fake_images = self.generator(z)
                fake_logits = self.discriminator(fake_images)
                g_loss = self.g_nonsat_loss(fake_logits)

                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                epoch_loss_D += d_loss.item()
                epoch_loss_G += g_loss.item()
            
            epoch_loss_D /= len(dataloader)
            epoch_loss_G /= len(dataloader)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | "
                      f"Loss_D: {epoch_loss_D:.4f} | Loss_G: {epoch_loss_G:.4f}")

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(z)
            generated_images = (generated_images.clamp(-1, 1) + 1) / 2.0
        return generated_images
    
    def __str__(self) -> str:
        return "StyleGAN"