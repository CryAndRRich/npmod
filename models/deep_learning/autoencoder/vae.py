import torch
import torch.nn as nn
from ..autoencoder import BaseAutoencoder

class VariationalAutoencoder(BaseAutoencoder):
    def __init__(self, 
                 input_dim: int = 784, 
                 latent_dim: int = 20, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.fc_mu = nn.Linear(in_features=128, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=128, out_features=latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128), 
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128), 
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=input_dim), 
            nn.Sigmoid()
        )

    def reparameterize(self, 
                       mu: float, 
                       logvar: float) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) to N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, 
                      x: torch.Tensor, 
                      recon_tuple: tuple) -> torch.Tensor:
        recon_x, mu, logvar = recon_tuple
    
        # Reconstruction loss 
        recon_loss = nn.functional.binary_cross_entropy(
            recon_x, x, reduction='mean' 
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss

    def __str__(self) -> str:
        return "Variational Autoencoder"