import torch
import torch.nn as nn
from ..autoencoder import BaseAutoencoder

class ContractiveAutoencoder(BaseAutoencoder):
    def __init__(self, 
                 input_dim: int = 784,
                 latent_dim: int =64, 
                 lam: float = 1e-4, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.lam = lam
        self.encoder_hidden = nn.Linear(in_features=input_dim, out_features=128)
        self.encoder_latent = nn.Linear(in_features=128, out_features=latent_dim)
        self.encoder_act = nn.Sigmoid()
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128), 
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=input_dim), 
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder_act(self.encoder_hidden(x)) # hidden layer output
        z = self.encoder_latent(h) # latent representation
        return h, z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, z = self.encode(x)
        return self.decoder(z)

    def loss_function(self, 
                      x: torch.Tensor, 
                      recon_x: torch.Tensor) -> torch.Tensor:
        mse = super().loss_function(x, recon_x)
        W = self.encoder_hidden.weight  
        h = self.encoder_act(self.encoder_hidden(x))   
        contractive_loss = torch.sum(
            (h * (1 - h))**2 @ torch.sum(W.pow(2), dim=1)
        )
        return mse + self.lam * contractive_loss

    def __str__(self) -> str:
        return "Contractive Autoencoder"