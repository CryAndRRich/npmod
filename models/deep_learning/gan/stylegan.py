import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..gan import GAN

class PixelNorm(nn.Module):
    def __init__(self) -> None:
        """
        Pixel-wise feature vector normalization
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize across channel dimension
        """
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class MappingNetwork(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 dlatent_dim: int = 512, 
                 num_layers: int = 8) -> None:
        """
        Mapping network: maps Z → W (intermediate latent space)

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
        """
        Forward pass: produce w from z
        """
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
        """
        Apply AdaIN to feature map x using style w

        Parameters:
            x: Feature map tensor (B, C, H, W)
            w: Style vector tensor (B, dlatent_dim)
        """
        B, C, _, _ = x.size()
        style_scale = self.style_scale(w).view(B, C, 1, 1)
        style_shift = self.style_shift(w).view(B, C, 1, 1)
        x_norm = self.norm(x)
        return style_scale * x_norm + style_shift

class StyledConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 dlatent_dim: int, 
                 upsample: bool = False) -> None:
        """
        Styled convolutional block with optional upsampling

        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dlatent_dim: Dimension of style vector w
            upsample: Whether to upsample by 2×
        """
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
        self.adain = AdaIN(out_channels, dlatent_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through styled convolution block

        Parameters:
            x: Feature map tensor
            w: Style vector tensor
        """
        if self.upsample:
            x = self.upsampler(x)
        x = self.conv(x)
        x = self.adain(x, w)
        x = self.lrelu(x)
        return x

class Generator(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 img_shape: Tuple[int], 
                 dlatent_dim: int = 512) -> None:
        """
        StyleGAN Generator

        Parameters:
            latent_dim: Dimension of input noise z
            img_shape: Shape of output image (C, H, W); assume H = W = 64, power of 2
            dlatent_dim: Dimension of intermediate latent vector w
        """
        super().__init__()
        C, H, _ = img_shape
        self.img_shape = img_shape
        self.mapping = MappingNetwork(latent_dim, dlatent_dim)
        # Start from constant input 4×4
        self.const_input = nn.Parameter(data=torch.randn(1, dlatent_dim, 4, 4))
        self.initial_conv = StyledConvBlock(dlatent_dim, dlatent_dim, dlatent_dim, upsample=False)
        # Upsample blocks to reach desired resolution
        self.to_rgb_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()
        num_layers = int(math.log2(H) - 2) 
        in_c, out_c = dlatent_dim, dlatent_dim
        for _ in range(num_layers):
            block = StyledConvBlock(in_c, out_c, dlatent_dim, upsample=True)
            self.blocks.append(block)
            self.to_rgb_layers.append(nn.Conv2d(in_channels=out_c, 
                                                out_channels=C, 
                                                kernel_size=1, 
                                                stride=1, 
                                                padding=0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of StyleGAN generator

        Parameters:
            z: Input noise tensor 
        """
        B = z.size(0)
        w = self.mapping(z)
        x = self.const_input.repeat(B, 1, 1, 1)
        x = self.initial_conv(x, w)
        for block, to_rgb in zip(self.blocks, self.to_rgb_layers):
            x = block(x, w)
        # Final ToRGB
        img = to_rgb(x)
        return torch.tanh(img)

class Discriminator(nn.Module):
    def __init__(self, img_shape: Tuple[int]) -> None:
        """
        Discriminator for StyleGAN

        Parameters:
            img_shape: Shape of input image (C, H, W), assume power-of-2 resolution
        """
        super().__init__()
        C, H, _ = img_shape
        self.model = nn.Sequential()
        in_c = C
        # Downsample blocks
        num_layers = int(math.log2(H) - 2)
        for i in range(num_layers):
            out_c = 512
            self.model.add_module(f"disc_conv_{i}", nn.Conv2d(in_channels=in_c, 
                                                              out_channels=out_c, 
                                                              kernel_size=4, 
                                                              stride=2, 
                                                              padding=1))
            self.model.add_module(f"disc_lrelu_{i}", nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_c = out_c
        # Final layers
        self.model.add_module("flatten", nn.Flatten())
        self.model.add_module("linear1", nn.Linear(in_features=512 * 4 * 4, out_features=1024))
        self.model.add_module("lrelu_final", nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.model.add_module("linear2", nn.Linear(in_features=1024, out_features=1))
        self.model.add_module("sigmoid", nn.Sigmoid())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator

        Parameters:
            img: Input image tensor

        Returns:
            Validity score (probability)
        """
        return self.model(img)

class StyleGAN(GAN):
    def init_network(self) -> None:
        """
        Initialize generator, discriminator, optimizers, and loss functions
        """
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.0, 0.99))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.0, 0.99))

        self.criterion = nn.BCELoss()

    def __str__(self) -> str:
        return "StyleGAN"
