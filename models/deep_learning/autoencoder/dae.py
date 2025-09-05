import torch
import torch.nn as nn
from ..autoencoder import BaseAutoencoder

class DenoisingAutoencoder(BaseAutoencoder):
    def __init__(self, 
                 input_dim: int = 784, 
                 latent_dim: int = 64, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128), 
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128), 
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=input_dim), 
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noisy = x + 0.2 * torch.randn_like(x)
        noisy = torch.clamp(noisy, 0., 1.)
        return self.decoder(self.encoder(noisy))
    
    def __str__(self) -> str:
        return "Denoising Autoencoder"

