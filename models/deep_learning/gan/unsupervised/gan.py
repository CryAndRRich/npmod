import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.deep_learning.gan import GAN

class Generator(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 img_shape: tuple) -> None:
        """
        Generator model of the GAN

        Parameters:
            latent_dim: Dimension of the noise vector
            img_shape: Shape of the output image
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=int(torch.prod(torch.tensor(self.img_shape)))),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator

        Parameters:
            z: Input noise tensor 

        Returns:
            generated_images: Tensor of generated images
        """
        img_flat = self.model(z)
        generated_images = img_flat.view(z.size(0), *self.img_shape)
        return generated_images

class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple) -> None:
        """
        Discriminator model of the GAN

        Parameters:
            img_shape: Shape of the input image 
        """
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=int(torch.prod(torch.tensor(self.img_shape))), out_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator

        Parameters:
            img: Input image tensor

        Returns:
            validity: Tensor representing real/fake probability
        """
        validity = self.model(img)
        return validity

class VanillaGAN(GAN):
    def init_network(self) -> None:
        """
        Initialize the generator, discriminator, optimizers, and loss function
        """
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate)
        self.criterion = nn.BCELoss()

    def fit(self, dataloader: DataLoader) -> None:
        self.init_network()
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.number_of_epochs):
            epoch_loss_G, epoch_loss_D = 0.0, 0.0

            for real_images, _ in dataloader:
                batch_size = real_images.size(0)
                
                real_label = 0.9
                eps = 0.05

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_validity = self.discriminator(real_images)
                valid = torch.empty_like(real_validity).uniform_(real_label - eps, real_label + eps)
                loss_real = self.criterion(real_validity, valid)

                z = torch.randn(batch_size, self.latent_dim)
                generated_images = self.generator(z)
                fake_validity = self.discriminator(generated_images.detach())
                fake = torch.empty_like(fake_validity).uniform_(0.0, eps)
                loss_fake = self.criterion(fake_validity, fake)

                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                gen_validity = self.discriminator(generated_images)
                loss_G = self.criterion(gen_validity, valid)
                loss_G.backward()
                self.optimizer_G.step()

                epoch_loss_D += loss_D.item()
                epoch_loss_G += loss_G.item()
            
            epoch_loss_D /= len(dataloader)
            epoch_loss_G /= len(dataloader)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] "
            #           f"Loss_D: {epoch_loss_D:.4f} | Loss_G: {epoch_loss_G:.4f}")

    def __str__(self) -> str:
        return "GAN (Generative Adversarial Network)"
