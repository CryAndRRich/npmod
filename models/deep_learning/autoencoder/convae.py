import torch
import torch.nn as nn
from ..autoencoder import BaseAutoencoder

class ConvAutoencoder(BaseAutoencoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=16, 
                      kernel_size=3, 
                      stride=2, 
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=2, 
                      padding=1), 
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, 
                               out_channels=16, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, 
                               out_channels=1, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  
            x = x.view(x.size(0), 1, 28, 28)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.view(out.size(0), -1)

    def __str__(self):
        return "Convolutional Autoencoder"