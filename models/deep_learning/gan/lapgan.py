from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ..gan import GAN

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
    def __init__(self, input_channels: int) -> None:
        """
        Discriminator for one level of LAPGAN

        Parameters:
            input_channels: Number of channels of the input image
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
            nn.Linear(in_features=128 * 7 * 7, out_features=1),
            nn.Sigmoid()
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
        C, _, _ = self.img_shape
        self.generators = nn.ModuleList()
        self.discriminators = nn.ModuleList()
        self.optimizers_G = []
        self.optimizers_D = []

        for _ in range(self.num_scales):
            # Each generator takes (upsampled image channels + noise channels) as input
            G = Generator(input_channels=C + C, output_channels=C)
            D = Discriminator(input_channels=C)

            G.apply(self.init_weights)
            D.apply(self.init_weights)

            self.generators.append(G)
            self.discriminators.append(D)

            self.optimizers_G.append(optim.Adam(G.parameters(), lr=self.learn_rate, betas=(0.5, 0.999)))
            self.optimizers_D.append(optim.Adam(D.parameters(), lr=self.learn_rate, betas=(0.5, 0.999)))

        self.criterion = nn.BCELoss()

    def fit(self, dataloader: DataLoader) -> None:
        """
        Train the LAPGAN using the provided DataLoader of real images.

        Parameters:
            dataloader: DataLoader yielding (real_images, _) tuples
        """
        self.init_network()
        C, _, _ = self.img_shape

        for _ in range(self.number_of_epochs):
            for real_images, _ in dataloader:
                # Real_images shape: (batch_size, C, H, W)
                batch_size = real_images.size(0)

                # Build Laplacian pyramid of real images
                current = real_images
                reals = [current]
                for _ in range(1, self.num_scales):
                    current = F.avg_pool2d(current, kernel_size=2)
                    reals.append(current)
                # Now reals = [full_res, half_res, quarter_res, ...]
                # We want to process from smallest to largest:
                reals = reals[::-1]  # [smallest, ..., full_res]

                # Iterate over each scale
                for scale in range(self.num_scales):
                    real = reals[scale]  # Real at this scale
                    G = self.generators[scale]
                    D = self.discriminators[scale]
                    opt_G = self.optimizers_G[scale]
                    opt_D = self.optimizers_D[scale]

                    # Labels for real/fake
                    valid = torch.ones(batch_size, 1)
                    fake = torch.zeros(batch_size, 1)

                    # Compute upsampled image from previous scale
                    if scale == 0:
                        # For the smallest scale, upsample from same size (just double)
                        upsampled = F.interpolate(real, scale_factor=2, mode='bilinear', align_corners=False)
                    else:
                        prev_real = reals[scale - 1]
                        upsampled = F.interpolate(prev_real, size=real.shape[2:], mode='bilinear', align_corners=False)

                    # Create noise tensor matching upsampled spatial dims
                    z = torch.randn(batch_size, C, upsampled.shape[2], upsampled.shape[3])
                    # Concatenate upsampled + noise as generator input
                    G_input = torch.cat((upsampled, z), dim=1)
                    generated_residual = G(G_input)
                    generated_image = generated_residual + upsampled

                    # Train Discriminator
                    opt_D.zero_grad()
                    real_validity = D(real)
                    fake_validity = D(generated_image.detach())
                    loss_real = self.criterion(real_validity, valid)
                    loss_fake = self.criterion(fake_validity, fake)
                    loss_D = (loss_real + loss_fake) / 2
                    loss_D.backward()
                    opt_D.step()

                    # Train Generator
                    opt_G.zero_grad()
                    gen_validity = D(generated_image)
                    loss_G = self.criterion(gen_validity, valid)
                    loss_G.backward()
                    opt_G.step()

    def __str__(self) -> str:
        return "Laplacian Pyramid GAN (LAPGAN)"
