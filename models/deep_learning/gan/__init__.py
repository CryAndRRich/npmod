from typing import Tuple
import torch
import torch.nn as nn

class GAN():
    def __init__(self,
                 latent_dim: int,
                 img_shape: Tuple[int, int, int],
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

    def init_weights(self, 
                     m: nn.Module,
                     type: str = "normal") -> None:
        """
        Initialize the model parameters

        Parameters:
            m: The module to initialize
            type: The type of initialization to use
        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if type == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif type == "orthogonal":
                nn.init.orthogonal_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def fit(self, 
            dataloader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        """
        Train the GAN using the provided DataLoader

        Parameters:
            dataloader: DataLoader for real images
            verbose: If True, print training progress
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

from .VanillaGAN import VanillaGAN
from .DCGAN import DCGAN
from .LAPGAN import LAPGAN
from .WGAN import WGAN
from .ProGAN import ProGAN
from .BigGAN import BigGAN
from .StyleGAN import StyleGAN