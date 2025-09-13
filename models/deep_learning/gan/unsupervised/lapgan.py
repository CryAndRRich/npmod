import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.deep_learning.gan import GAN

class Generator(nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 output_channels: int) -> None:
        """
        Generator for one level of LAPGAN

        Parameters:
            input_channels: Number of channels of (upsampled image + noise)
            output_channels: Number of channels of the output image
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, 
                      out_channels=output_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator

        Parameters:
            x: Concatenated tensor 

        Returns:
            Residual image at this scale
        """
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 img_size: int) -> None:
        """
        Discriminator for one level of LAPGAN

        Parameters:
            input_channels: Number of channels of the input image
            img_size: Height/Width of the input image (assumed square)
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=2, 
                      padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=3, 
                      stride=2, 
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=128 * (img_size // 4) * (img_size // 4), out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator

        Parameters:
            x: Input image tensor

        Returns:
            Validity score (real/fake probability)
        """
        return self.model(x)

class LAPGAN(GAN):
    def __init__(self,
                 latent_dim: int,
                 img_shape: Tuple[int],
                 learn_rate: float,
                 number_of_epochs: int,
                 num_scales: int = 3) -> None:
        """
        Laplacian Pyramid GAN (LAPGAN)

        Parameters:
            latent_dim: Dimension of the noise vector per channel
            img_shape: Shape of the images to generate (C, H, W)
            learn_rate: Learning rate for all optimizers
            number_of_epochs: Number of training epochs
            num_scales: Number of pyramid levels (from smallest to full resolution)
        """
        super().__init__(latent_dim, img_shape, learn_rate, number_of_epochs)
        self.num_scales = num_scales

    def init_network(self) -> None:
        """
        Initialize Generator and Discriminator for each pyramid level,
        their weights, optimizers, and the loss function.
        """
        C, H, _ = self.img_shape
        img_sizes = [H // (2 ** (self.num_scales - 1 - i)) for i in range(self.num_scales)]

        self.generators = nn.ModuleList()
        self.discriminators = nn.ModuleList()
        self.optimizers_G = []
        self.optimizers_D = []

        for size in img_sizes:
            G = Generator(input_channels=C + self.latent_dim, output_channels=C) 
            D = Discriminator(input_channels=C, img_size=size)

            G.apply(self.init_weights)
            D.apply(self.init_weights)

            self.generators.append(G)
            self.discriminators.append(D)

            self.optimizers_G.append(optim.Adam(G.parameters(), lr=self.learn_rate, betas=(0.5, 0.999)))
            self.optimizers_D.append(optim.Adam(D.parameters(), lr=self.learn_rate * 0.5, betas=(0.5, 0.999)))

        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, dataloader: DataLoader) -> None:
        self.init_network()
        for G, D in zip(self.generators, self.discriminators):
            G.train()
            D.train()

        C, _, _ = self.img_shape

        for epoch in range(self.number_of_epochs):
            epoch_loss_G, epoch_loss_D = 0.0, 0.0

            for real_images, _ in dataloader:
                batch_size = real_images.size(0)

                # Build Laplacian pyramid of real images
                current = real_images
                reals = [current]
                for _ in range(1, self.num_scales):
                    current = F.avg_pool2d(current, kernel_size=2)
                    reals.append(current)

                reals = reals[::-1]  # [smallest, ..., full_res]

                loss_D_total, loss_G_total = 0.0, 0.0
                # Iterate over each scale
                for scale in range(self.num_scales):
                    real = reals[scale]  # Real at this scale
                    G = self.generators[scale]
                    D = self.discriminators[scale]
                    opt_G = self.optimizers_G[scale]
                    opt_D = self.optimizers_D[scale]

                    # Compute upsampled image from previous scale
                    if scale == 0:
                        upsampled = torch.randn(batch_size, C, real.shape[2], real.shape[3])
                    else:
                        prev_real = reals[scale - 1]
                        upsampled = F.interpolate(prev_real, size=real.shape[2:], mode="bilinear", align_corners=False)

                    # Create noise tensor matching upsampled spatial dims
                    z = torch.randn(batch_size, self.latent_dim, upsampled.shape[2], upsampled.shape[3])
                    # Concatenate upsampled + noise as generator input
                    G_input = torch.cat((upsampled, z), dim=1)
                    generated_residual = G(G_input)
                    generated_image = generated_residual + upsampled

                    # Train Discriminator
                    opt_D.zero_grad()
                    eps = 0.05
                    real_label = 0.9


                    real_validity = D(real)
                    fake_validity = D(generated_image.detach())
                    # Labels for real/fake
                    valid = torch.empty_like(real_validity).uniform_(real_label - eps, real_label + eps)
                    fake = torch.empty_like(fake_validity).uniform_(0.0, eps)

                    loss_real = self.criterion(real_validity, valid)
                    loss_fake = self.criterion(fake_validity, fake)
                    loss_D = 0.5 * (loss_real + loss_fake)
                    loss_D.backward()
                    opt_D.step()

                    opt_G.zero_grad()
                    gen_validity = D(generated_image)
                    target_for_G = torch.ones_like(gen_validity) * real_label
                    loss_G = self.criterion(gen_validity, target_for_G)
                    loss_G.backward()
                    opt_G.step()

                    loss_D_total += loss_D.item()
                    loss_G_total += loss_G.item()

                epoch_loss_D += loss_D_total / self.num_scales
                epoch_loss_G += loss_G_total / self.num_scales
            
            epoch_loss_D /= len(dataloader)
            epoch_loss_G /= len(dataloader)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] "
                      f"Loss_D: {epoch_loss_D:.4f} | Loss_G: {epoch_loss_G:.4f}")

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate images from random noise using the trained LAPGAN.

        Parameters:
            batch_size: Number of images to generate

        Returns:
            generated_images: Tensor of generated images at final resolution
        """
        C, H, _ = self.img_shape
        for G in self.generators:
            G.eval()

        with torch.no_grad():
            smallest = H // (2 ** (self.num_scales - 1))
            z0 = torch.randn(batch_size, self.latent_dim, smallest, smallest)
            upsampled = torch.zeros(batch_size, C, smallest, smallest)
            G_input = torch.cat((upsampled, z0), dim=1)
            current_image = self.generators[0](G_input) + upsampled

        with torch.no_grad():
            smallest = H // (2 ** (self.num_scales - 1))
            z0 = torch.randn(batch_size, self.latent_dim, smallest, smallest)
            upsampled = torch.zeros(batch_size, C, smallest, smallest)
            G_input = torch.cat((upsampled, z0), dim=1)
            current_image = self.generators[0](G_input) + upsampled

            for scale in range(1, self.num_scales):
                target_size = H // (2 ** (self.num_scales - 1 - scale))
                upsampled = F.interpolate(current_image, size=(target_size, target_size), mode="bilinear", align_corners=False)
                z = torch.randn(batch_size, self.latent_dim, target_size, target_size)
                G_input = torch.cat((upsampled, z), dim=1)
                residual = self.generators[scale](G_input)
                current_image = upsampled + residual

            current_image = (current_image.clamp(-1,1) + 1) / 2

        return current_image

    def __str__(self) -> str:
        return "LAPGAN (Laplacian Pyramid GAN)"
