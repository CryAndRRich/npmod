from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from ..gan import GAN

class Generator(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 img_shape: Tuple[int]) -> None:
        """
        Simple Generator for WGAN

        Parameters:
            latent_dim: Dimension of the input noise vector
            img_shape: Shape of output image
        """
        super().__init__()
        C, _, _ = img_shape
        self.img_shape = img_shape
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
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, img_shape: Tuple[int]) -> None:
        """
        Critic (discriminator) for WGAN, without sigmoid

        Parameters:
            img_shape: Shape of input image
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
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=256, 
                      out_channels=512, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1),
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
        return self.model(img)

class WGAN(GAN):
    def __init__(self,
                 latent_dim: int,
                 img_shape: Tuple[int, int, int],
                 learn_rate: float,
                 number_of_epochs: int,
                 critic_iters: int = 5,
                 lambda_gp: float = 10.0) -> None:
        """
        Wasserstein GAN with gradient penalty

        Parameters:
            latent_dim: Dimension of the noise vector
            img_shape: Shape of generated images (C, H, W)
            learn_rate: Learning rate for both generator and critic
            number_of_epochs: Number of epochs to train
            critic_iters: How many critic updates per generator update
            lambda_gp: Gradient penalty coefficient
        """
        super().__init__(
            latent_dim=latent_dim, 
            img_shape=img_shape, 
            learn_rate=learn_rate, 
            number_of_epochs=number_of_epochs
        )
        self.critic_iters = critic_iters
        self.lambda_gp = lambda_gp

    def init_network(self) -> None:
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.critic = Critic(self.img_shape)

        self.generator.apply(self.init_weights)
        self.critic.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.0, 0.9))
        self.optimizer_C = optim.Adam(self.critic.parameters(), lr=self.learn_rate, betas=(0.0, 0.9))

    @staticmethod
    def compute_gradient_penalty(critic: nn.Module, 
                                 real_samples: torch.Tensor, 
                                 fake_samples: torch.Tensor, 
                                 lambda_gp: float = 10.0) -> torch.Tensor:
        """
        Compute the gradient penalty for WGAN-GP
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_samples)

        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        critic_interpolates = critic(interpolates)
        grad_outputs = torch.ones_like(critic_interpolates)

        gradients = autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  

        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        penalty = ((gradients_norm - 1) ** 2).mean() * lambda_gp
        return penalty

    def fit(self, 
            dataloader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
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

                    real_validity = self.critic(real_images)
                    fake_validity = self.critic(fake_images)

                    # WGAN critic loss
                    loss_C = fake_validity.mean() - real_validity.mean()
                    
                    # Gradient penalty
                    gp = self.compute_gradient_penalty(self.critic, real_images, fake_images, self.lambda_gp)
                    
                    loss_C_total_step = loss_C + gp
                    loss_C_total_step.backward()
                    self.optimizer_C.step()

                    loss_C_total += loss_C_total_step.item()

                # Train Generator
                z = torch.randn(batch_size, self.latent_dim, 1, 1)
                self.optimizer_G.zero_grad()
                gen_images = self.generator(z)

                gen_validity = self.critic(gen_images)
                loss_G = -gen_validity.mean()
                loss_G.backward()
                self.optimizer_G.step()

                epoch_loss_C += loss_C_total / self.critic_iters
                epoch_loss_G += loss_G.item()
            
            epoch_loss_C /= len(dataloader)
            epoch_loss_G /= len(dataloader)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | "
                      f"Loss_C: {epoch_loss_C:.4f} | Loss_G: {epoch_loss_G:.4f}")
                
    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(noise)
            generated_images = (generated_images.clamp(-1, 1) + 1) / 2.0
        return generated_images

    def __str__(self) -> str:
        return "Wasserstein GAN (WGAN-GP)"