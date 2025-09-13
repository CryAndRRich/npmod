import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.deep_learning.gan import GAN

class Generator(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 img_shape: Tuple[int]) -> None:
        """
        Simple Generator for WGAN

        Parameters:
            latent_dim: Dimension of the input noise vector
            img_shape: Shape of output image (C, H, W)
        """
        super().__init__()
        C, _, _ = img_shape
        self.img_shape = img_shape
        # We'll upsample from 1×1 to H×W with transposed convolutions
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, 
                               out_channels=512, 
                               kernel_size=4, 
                               stride=1, 
                               padding=0, 
                               bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=512, 
                               out_channels=256, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=256, 
                               out_channels=128, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=64, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, 
                               out_channels=C, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator

        Parameters:
            z: Noise tensor 

        Returns:
            Generated image tensor 
        """
        img = self.model(z)
        return img

class Critic(nn.Module):
    def __init__(self, img_shape: Tuple[int]) -> None:
        """
        Critic (discriminator) for WGAN, without sigmoid

        Parameters:
            img_shape: Shape of input image (C, H, W)
        """
        super().__init__()
        C, H, W = img_shape
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=C, 
                      out_channels=64, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=256, 
                      out_channels=512, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=512 * (H // 16) * (W // 16), out_features=1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic

        Parameters:
            img: Input image tensor

        Returns:
            Critic score (realness)
        """
        validity = self.model(img)
        return validity

class WGAN(GAN):
    def __init__(self,
                 latent_dim: int,
                 img_shape: Tuple[int],
                 learn_rate: float,
                 number_of_epochs: int,
                 critic_iters: int = 5,
                 clip_value: float = 0.01) -> None:
        """
        Wasserstein GAN (WGAN) with weight clipping

        Parameters:
            latent_dim: Dimension of the noise vector
            img_shape: Shape of generated images (C, H, W)
            learn_rate: Learning rate for both generator and critic
            number_of_epochs: Number of epochs to train
            critic_iters: How many critic updates per generator update
            clip_value: Weight clipping threshold
        """
        super().__init__(latent_dim, img_shape, learn_rate, number_of_epochs)
        self.critic_iters = critic_iters
        self.clip_value = clip_value

    def init_network(self) -> None:
        """
        Initialize generator, critic, their optimizers, and the loss function
        """
        # Instantiate generator and critic
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.critic = Critic(self.img_shape)

        # Apply weight initialization
        self.generator.apply(self.init_weights)
        self.critic.apply(self.init_weights)

        # Optimizers: RMSprop as recommended in original WGAN paper
        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=self.learn_rate)
        self.optimizer_C = optim.RMSprop(self.critic.parameters(), lr=self.learn_rate)

        # No explicit loss function object, WGAN loss uses raw critic outputs

    def fit(self, dataloader: DataLoader) -> None:
        """
        Train the WGAN using the provided DataLoader

        Parameters:
            dataloader: DataLoader yielding (real_images, _) tuples
        """
        self.init_network()
        self.generator.train()
        self.critic.train()

        for epoch in range(self.number_of_epochs):
            epoch_loss_G, epoch_loss_C = 0.0, 0.0
            
            for real_images, _ in dataloader:
                batch_size = real_images.size(0)

                # Train Critic
                loss_C_total = 0.0
                for _ in range(self.critic_iters):
                    z = torch.randn(batch_size, self.latent_dim, 1, 1)
                    fake_images = self.generator(z).detach()
                    
                    self.optimizer_C.zero_grad()
                    # Critic outputs
                    real_validity = self.critic(real_images)
                    fake_validity = self.critic(fake_images)
                    # WGAN critic loss = E[fake] - E[real]
                    loss_C = fake_validity.mean() - real_validity.mean()
                    loss_C.backward()
                    self.optimizer_C.step()

                    # Weight clipping
                    for p in self.critic.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                    loss_C_total += loss_C.item()

                # Train Generator
                z = torch.randn(batch_size, self.latent_dim, 1, 1)
                self.optimizer_G.zero_grad()
                gen_images = self.generator(z)
                # Generator loss = -E[critic(gen_images)]
                gen_validity = self.critic(gen_images)
                loss_G = -gen_validity.mean()
                loss_G.backward()
                self.optimizer_G.step()

                epoch_loss_C += loss_C_total / self.critic_iters
                epoch_loss_G += loss_G.item()
            
            epoch_loss_C /= len(dataloader)
            epoch_loss_G /= len(dataloader)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] "
                      f"Loss_D: {epoch_loss_C:.4f} | Loss_G: {epoch_loss_G:.4f}")
                
    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        noise = noise.to(next(self.generator.parameters()).device)
        with torch.no_grad():
            generated_images = self.generator(noise)
            generated_images = (generated_images.clamp(-1,1) + 1) / 2
        return generated_images

    def __str__(self) -> str:
        return "Wasserstein GAN (WGAN)"
