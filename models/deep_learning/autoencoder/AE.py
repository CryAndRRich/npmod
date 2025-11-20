from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from ..autoencoder import Autoencoder

class Encoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 latent_dim: int, 
                 hidden_dims: List[int] = [128, 64]) -> None:
        """
        Fully Connected Encoder
        
        Parameters:
            input_shape: Shape of the input data (C, H, W)
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden layer dimensions
        """
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
            
        networks.append(nn.Linear(in_dim, latent_dim))
        
        self.net = nn.Sequential(*networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 output_shape: Tuple[int, int, int], 
                 hidden_dims: List[int] = [128, 64]) -> None:
        """
        Fully Connected Decoder
        
        Parameters:
            latent_dim: Dimension of the latent space
            output_shape: Shape of the output data (C, H, W)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        C, H, W = output_shape
        self.flat_dim = C * H * W
        
        hidden_dims = hidden_dims[::-1]
        networks = []
        
        in_dim = latent_dim
        for h_dim in hidden_dims:
            networks.append(nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=h_dim),
                nn.BatchNorm1d(num_features=h_dim),
                nn.ReLU()
            ))
            in_dim = h_dim
            
        networks.append(nn.Linear(in_features=in_dim, out_features=self.flat_dim))
        
        networks.append(nn.Sigmoid())
        networks.append(nn.Unflatten(dim=1, unflattened_size=(C, H, W)))
        
        self.net = nn.Sequential(*networks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class VanillaAE(Autoencoder):
    def __init__(self,
                 latent_dim: int,
                 input_shape: Tuple[int, int, int],
                 learn_rate: float = 1e-3,
                 number_of_epochs: int = 50,
                 hidden_dims: List[int] = [128, 64]) -> None:
        super().__init__(
            latent_dim=latent_dim, 
            input_shape=input_shape, 
            learn_rate=learn_rate, 
            number_of_epochs=number_of_epochs
        )
        self.hidden_dims = hidden_dims

    def init_network(self) -> None:
        self.encoder = Encoder(self.input_shape, self.latent_dim, self.hidden_dims)
        self.decoder = Decoder(self.latent_dim, self.input_shape, self.hidden_dims)

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=self.learn_rate)
        self.criterion = nn.MSELoss()

    def fit(self, 
            dataloader: torch.utils.data.DataLoader, 
            verbose: bool = False) -> None:
        self.init_network()
        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.number_of_epochs):
            total_loss = 0.0
            for img, _ in dataloader:
                z = self.encoder(img)
                reconstruction = self.decoder(z)

                loss = self.criterion(reconstruction, img)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | Loss: {total_loss / len(dataloader):.6f}")
        
    def __str__(self) -> str:
        return "Autoencoder"