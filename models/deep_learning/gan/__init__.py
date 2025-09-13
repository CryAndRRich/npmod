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
        Initialize the model parameters

        Parameters:
            m: The module to initialize
        """
        if hasattr(m, 'weight') and m.weight is not None:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def fit(self, dataloader: DataLoader) -> None:
        """
        Train the GAN using the provided DataLoader

        Parameters:
            dataloader: DataLoader for real images
        """
        pass

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

from .supervised import *
from .unsupervised import *