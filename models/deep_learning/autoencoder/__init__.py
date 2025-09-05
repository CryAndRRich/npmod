import torch
import torch.nn as nn
import torch.optim as optim

class BaseAutoencoder(nn.Module):
    def __init__(self, 
                 learn_rate: float = 1e-3, 
                 number_of_epochs: int = 20) -> None:
        super().__init__()
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        
        # Placeholders for encoder and decoder
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

        # Optimizer and loss function
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder"""
        return self.decoder(self.encoder(x))

    def loss_function(self, x: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss"""
        return self.criterion(recon_x, x)

    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Train the autoencoder
        
        Parameters:
            dataloader: DataLoader for training data
        """
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        self.train()
        for _ in range(self.number_of_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                recon = self.forward(batch)
                loss = self.loss_function(batch, recon)
                loss.backward()
                self.optimizer.step()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded representation of the input data"""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)
        
    def __str__(self) -> str:
        pass
    

from .dae import DenoisingAutoencoder
from .vae import VariationalAutoencoder
from .cae import ContractiveAutoencoder
from .convae import ConvAutoencoder