from typing import Tuple
import torch
import torch.nn as nn

class Autoencoder():
    def __init__(self,
                 latent_dim: int,
                 input_shape: Tuple[int, int, int],
                 learn_rate: float,
                 number_of_epochs: int) -> None:
        """
        Initializes the Autoencoder wrapper

        Parameters:
            latent_dim: Dimension of the bottleneck (latent space)
            input_shape: Shape of the input data (C, H, W)
            learn_rate: Learning rate for the optimizer
            number_of_epochs: Number of training iterations
        """
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs

    def init_network(self) -> None:
        """
        Initialize the encoder, decoder, optimizers, and loss function
        """
        pass

    def init_weights(self, m: nn.Module) -> None:
        """
        Initialize the model parameters
        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def fit(self, 
            dataloader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        """
        Train the Autoencoder using the provided DataLoader

        Parameters:
            dataloader: DataLoader for training data
            verbose: If True, print training progress
        """
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input data to the latent space

        Parameters:
            x: Input tensor

        Returns:
            z: Latent representation
        """
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map latent vector back to data space

        Parameters:
            z: Latent tensor

        Returns:
            x_hat: Reconstructed data
        """
        self.decoder.eval()
        with torch.no_grad():
            x_hat = self.decoder(z)
        return x_hat

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input data through the autoencoder

        Parameters:
            x: Input tensor

        Returns:
            x_hat: Reconstructed data
        """
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            z = self.encoder(x)
            x_hat = self.decoder(z)
        return x_hat

from .AE import VanillaAE
from .RegularizedAE import RegularizedAE
from .ConvolutionalAE import ConvolutionalAE
from .VAE import VAE
from .AAE import AAE
from .VQVAE import VQVAE
from .MAE import MAE