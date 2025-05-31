from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class GAN():
    def __init__(self,
                 latent_dim: int,
                 img_shape: Tuple[int],
                 learn_rate: float,
                 number_of_epochs: int) -> None:
        """
        Initializes the GAN with generator and discriminator

        Parameters:
            latent_dim: Dimension of the noise vector
            img_shape: Shape of the images to generate 
            learn_rate: Learning rate for both generator and discriminator
            number_of_epochs: Number of training iterations
        """
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs

    def init_network(self) -> None:
        """
        Initialize the generator, discriminator, optimizers, and loss function
        """
        pass

    def init_weights(self, m: nn.Module) -> None:
        """
        Initialize the model parameters using the Xavier initializer

        Parameters:
            m: The module to initialize
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def fit(self, dataloader: DataLoader) -> None:
        """
        Train the GAN using the provided DataLoader

        Parameters:
            dataloader: DataLoader for real images
        """
        self.init_network()

        for _ in range(self.number_of_epochs):
            for real_images, _ in dataloader:
                batch_size = real_images.size(0)

                valid = torch.ones((batch_size, 1))
                fake = torch.zeros((batch_size, 1))

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_validity = self.discriminator(real_images)
                loss_real = self.criterion(real_validity, valid)

                z = torch.randn(batch_size, self.latent_dim)
                generated_images = self.generator(z)
                fake_validity = self.discriminator(generated_images.detach())
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

    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate images from noise vectors using the trained generator

        Parameters:
            noise: Input noise tensor

        Returns:
            generated_images: Tensor of generated images
        """
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(noise)
        return generated_images

from .gan import VanillaGAN
from .dcgan import DCGAN
from .lapgan import LAPGAN
from .srgan import SRGAN
from .wgan import WGAN
from .cgan import CGAN
from .pix2pix import Pix2Pix
from .stylegan import StyleGAN
from .spade import SPADE