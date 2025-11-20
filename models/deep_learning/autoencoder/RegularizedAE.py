from typing import List, Tuple
import torch
import torch.nn as nn
from .AE import Encoder as VanillaEncoder, Decoder as VanillaDecoder, VanillaAE

class Encoder(VanillaEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net.add_module("latent_activation", nn.Sigmoid())

class Decoder(VanillaDecoder):
    pass

class RegularizedAE(VanillaAE):
    def __init__(self,
                 latent_dim: int,
                 input_shape: Tuple[int, int, int],
                 learn_rate: float = 1e-3,
                 number_of_epochs: int = 50,
                 hidden_dims: List[int] = [128, 64],
                 reg_type: str = "denoising", 
                 reg_coeff: float = 1e-5,
                 noise_factor: float = 0.5) -> None:
        """
        Regularized Autoencoder supporting Denoising, Sparse, and Contractive variants
        
        Parameters:
            latent_dim: Dimension of the latent space
            input_shape: Shape of the input data (C, H, W)
            learn_rate: Learning rate for the optimizer
            number_of_epochs: Number of training epochs
            hidden_dims: List of hidden layer dimensions
            reg_type: Type of regularization ("denoising", "sparse", "contractive")
            reg_coeff: Regularization coefficient
            noise_factor: Noise factor for denoising autoencoder
        """
        super().__init__(
            latent_dim=latent_dim,
            input_shape=input_shape,
            learn_rate=learn_rate,
            number_of_epochs=number_of_epochs,
            hidden_dims=hidden_dims
        )
        
        assert reg_type in ["denoising", "sparse", "contractive"], "Invalid reg_type"
        self.reg_type = reg_type
        self.reg_coeff = reg_coeff
        self.noise_factor = noise_factor

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.noise_factor
        return torch.clamp(x + noise, 0., 1.)

    def _sparse_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(z))

    def _contractive_loss(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gradients = torch.autograd.grad(
            outputs=z.sum(), 
            inputs=x, 
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        return torch.sum(gradients.pow(2)) / x.size(0)

    def fit(self, 
            dataloader: torch.utils.data.DataLoader, 
            verbose: bool = False) -> None:
        self.init_network()
        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.number_of_epochs):
            total_loss = 0.0
            for img, _ in dataloader:
                if self.reg_type == "denoising":
                    input_x = self._add_noise(img)
                else:
                    input_x = img

                if self.reg_type == "contractive":
                    input_x.requires_grad_(True)
                
                z = self.encoder(input_x)
                reconstruction = self.decoder(z)

                loss = self.criterion(reconstruction, img)

                if self.reg_type == "sparse":
                    loss += self.reg_coeff * self._sparse_loss(z)
                elif self.reg_type == "contractive":
                    loss += self.reg_coeff * self._contractive_loss(z, input_x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | Loss: {total_loss / len(dataloader):.6f}")

    def __str__(self) -> str:
        return "Regularized Autoencoder"