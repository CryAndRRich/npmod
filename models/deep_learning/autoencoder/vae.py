from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from ..autoencoder import Autoencoder
from .AE import Decoder as VanillaDecoder

class Encoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 latent_dim: int, 
                 hidden_dims: List[int] = [128, 64]) -> None:
        super().__init__()
        
        C, H, W = input_shape
        self.flat_dim = C * H * W
        
        networks = []
        networks.append(nn.Flatten())
        
        in_dim = self.flat_dim
        for h_dim in hidden_dims:
            networks.append(nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=h_dim),
                nn.BatchNorm1d(num_features=h_dim),
                nn.ReLU()
            ))
            in_dim = h_dim
        
        self.net = nn.Sequential(*networks)
        
        self.fc_mu = nn.Linear(in_dim, latent_dim)      
        self.fc_var = nn.Linear(in_dim, latent_dim)     

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
class Decoder(VanillaDecoder):
    pass

class VAE(Autoencoder):
    def __init__(self,
                 latent_dim: int,
                 input_shape: Tuple[int, int, int],
                 learn_rate: float = 1e-3,
                 number_of_epochs: int = 50,
                 hidden_dims: List[int] = [128, 64],
                 kld_weight: float = 1.0) -> None:
        """
        Variational Autoencoder
        
        Parameters:
            latent_dim: Dimension of the latent space
            input_shape: Shape of the input data (C, H, W)
            learn_rate: Learning rate for the optimizer
            number_of_epochs: Number of training epochs
            hidden_dims: List of hidden layer dimensions
            kld_weight: Weight for the Kullback-Leibler Divergence loss
        """
        super().__init__(
            latent_dim=latent_dim, 
            input_shape=input_shape, 
            learn_rate=learn_rate, 
            number_of_epochs=number_of_epochs
        )
        self.hidden_dims = hidden_dims
        self.kld_weight = kld_weight

    def init_network(self) -> None:
        self.encoder = Encoder(self.input_shape, self.latent_dim, self.hidden_dims)
        self.decoder = Decoder(self.latent_dim, self.input_shape, self.hidden_dims)

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=self.learn_rate)
        self.criterion = nn.MSELoss(reduction="sum") 

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def fit(self, 
            dataloader: torch.utils.data.DataLoader, 
            verbose: bool = False) -> None:
        self.init_network()
        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.number_of_epochs):
            total_loss = 0.0
            total_kld = 0.0
            total_recon = 0.0

            for img, _ in dataloader:
                mu, log_var = self.encoder(img)

                z = self.reparameterize(mu, log_var)
                reconstruction = self.decoder(z)

                recon_loss = self.criterion(reconstruction, img)
                
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                loss = recon_loss + (self.kld_weight * kld_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()

            avg_loss = total_loss / len(dataloader.dataset)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | "
                      f"Total Loss: {avg_loss:.4f} "
                      f"(Recon: {total_recon/len(dataloader.dataset):.4f}, "
                      f"KLD: {total_kld/len(dataloader.dataset):.4f})")
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generate new images by sampling from Normal Distribution
        """
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            samples = self.decoder(z)
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            mu, log_var = self.encoder(x)
            z = self.reparameterize(mu, log_var)
            x_hat = self.decoder(z)
        return x_hat

    def __str__(self) -> str:
        return "Variational Autoencoder (VAE)"