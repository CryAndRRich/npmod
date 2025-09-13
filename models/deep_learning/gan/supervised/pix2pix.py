import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.deep_learning.gan import GAN

class UNetDown(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 batch_norm: bool = True) -> None:
        """
        Down-sampling block for U-Net

        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            batch_norm: Whether to use BatchNorm
        """
        super().__init__()
        layers = [nn.Conv2d(in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=4, 
                            stride=2, 
                            padding=1, 
                            bias=False)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 dropout: bool = False) -> None:
        """
        Up-sampling block for U-Net

        Parameters:
            in_channels: Number of input channels (from previous layer)
            out_channels: Number of output channels
            dropout: Whether to apply dropout
        """
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(p=0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, 
                x: torch.Tensor, 
                skip_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection

        Parameters:
            x: Input feature map (bottleneck or previous upsample)
            skip_input: Corresponding feature map from down path
        """
        x = self.model(x)
        # Concatenate before feeding to next block
        return torch.cat((x, skip_input), 1)

class GeneratorUNet(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3) -> None:
        """
        U-Net Generator for Pix2Pix

        Parameters:
            in_channels: Number of channels in input image 
            out_channels: Number of channels in output image
        """
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, batch_norm=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, batch_norm=False)

        self.up1 = UNetUp(512, 512, dropout=True)
        self.up2 = UNetUp(1024, 512, dropout=True)
        self.up3 = UNetUp(1024, 512, dropout=True)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=out_channels, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net generator

        Parameters:
            x: Input image tensor (low-resolution or source domain)

        Returns:
            Generated image tensor (translated to target domain)
        """
        # Downsampling path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Upsampling path with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class DiscriminatorPix2Pix(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        """
        PatchGAN Discriminator for Pix2Pix

        Parameters:
            in_channels: Number of channels of input images (concatenated input + target)
        """
        super().__init__()
        def disc_block(in_c: int, 
                       out_c: int, 
                       stride: int, 
                       batch_norm: bool) -> nn.Sequential:
            layers = [nn.Conv2d(in_channels=in_c, 
                                out_channels=out_c, 
                                kernel_size=4, 
                                stride=stride, 
                                padding=1)
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_features=out_c))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return nn.Sequential(*layers)

        # Input: concatenated (source, target) → 6 channels if RGB
        self.model = nn.Sequential(
            disc_block(in_channels * 2, 64, stride=2, batch_norm=False),
            disc_block(64, 128, stride=2, batch_norm=True),
            disc_block(128, 256, stride=2, batch_norm=True),
            disc_block(256, 512, stride=1, batch_norm=True),
            nn.Conv2d(in_channels=512, 
                      out_channels=1, 
                      kernel_size=4, 
                      stride=1, 
                      padding=1),  # Patch output
            nn.Sigmoid()
        )

    def forward(self, 
                input_img: torch.Tensor, 
                target_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchGAN discriminator

        Parameters:
            input_img: Source (condition) image tensor
            target_img: Real or generated target image tensor

        Returns:
            Patch-wise real/fake probabilities
        """
        x = torch.cat((input_img, target_img), 1)
        return self.model(x)

class Pix2Pix(GAN):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 learn_rate: float,
                 number_of_epochs: int,
                 lambda_l1: float = 100.0) -> None:
        """
        Pix2Pix GAN for image-to-image translation

        Parameters:
            in_channels: Number of channels in input image
            out_channels: Number of channels in output image
            learn_rate: Learning rate for optimizers
            number_of_epochs: Number of epochs to train
            lambda_l1: Weight for L1 pixel-wise loss
        """
        # latent_dim unused for Pix2Pix; use GAN base for structure
        super().__init__(latent_dim=None, img_shape=(out_channels,), learn_rate=learn_rate, number_of_epochs=number_of_epochs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lambda_l1 = lambda_l1

    def init_network(self) -> None:
        """
        Initialize U-Net generator, PatchGAN discriminator, optimizers, and loss functions
        """
        self.generator = GeneratorUNet(in_channels=self.in_channels, out_channels=self.out_channels)
        self.discriminator = DiscriminatorPix2Pix(in_channels=self.in_channels)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))

        self.criterion_GAN = nn.BCELoss()
        self.criterion_L1 = nn.L1Loss()

    def fit(self, dataloader: DataLoader) -> None:
        """
        Train the Pix2Pix GAN using paired data

        Parameters:
            dataloader: DataLoader yielding (input_img, target_img) pairs
        """
        self.init_network()

        for _ in range(self.number_of_epochs):
            for input_img, target_img in dataloader:
                batch_size = input_img.size(0)
                # Real and fake labels
                valid = torch.ones((batch_size, 1, 30, 30))
                fake = torch.zeros((batch_size, 1, 30, 30))

                # Train Discriminator
                self.optimizer_D.zero_grad()
                # Real pairs
                pred_real = self.discriminator(input_img, target_img)
                loss_D_real = self.criterion_GAN(pred_real, valid)
                # Generated pairs
                gen_img = self.generator(input_img)
                pred_fake = self.discriminator(input_img, gen_img.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, fake)
                # Total D loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                # Adversarial loss
                pred_fake_for_G = self.discriminator(input_img, gen_img)
                loss_G_GAN = self.criterion_GAN(pred_fake_for_G, valid)
                # L1 loss
                loss_G_L1 = self.criterion_L1(gen_img, target_img) * self.lambda_l1
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                self.optimizer_G.step()

    def __str__(self) -> str:
        return "Pix2Pix GAN"
