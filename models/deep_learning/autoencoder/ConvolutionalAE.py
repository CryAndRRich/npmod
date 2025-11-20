from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from ..autoencoder import Autoencoder

class Encoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 latent_dim: int, 
                 hidden_channels: List[int] = [32, 64, 128]) -> None:
        """
        Convolutional Encoder

        Parameters:
            input_shape: Shape of the input data (C, H, W)
            latent_dim: Dimension of the latent space
            hidden_channels: List of channels for Conv layers
        """
        super().__init__()
        
        self.input_shape = input_shape
        C, _, _ = input_shape
        
        networks = []
        in_channels = C
        
        for h_dim in hidden_channels:
            networks.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=h_dim, 
                          kernel_size=3, 
                          stride=2, 
                          padding=1),
                nn.BatchNorm2d(num_features=h_dim),
                nn.ReLU()
            ))
            in_channels = h_dim
            
        self.net = nn.Sequential(*networks)
        
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.net(dummy)
            self.shape_before_flatten = conv_out.shape[1:]
            self.flat_dim = conv_out.view(1, -1).size(1)
            
        self.fc = nn.Linear(in_features=self.flat_dim, out_features=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 output_shape: Tuple[int, int, int], 
                 shape_before_flatten: Tuple[int, int, int],
                 hidden_channels: List[int] = [32, 64, 128]) -> None:
        """
        Convolutional Decoder

        Parameters:
            latent_dim: Dimension of the latent space
            output_shape: Shape of the output data (C, H, W)
            shape_before_flatten: Shape of the feature map before flattening in Encoder
            hidden_channels: List of channels for Conv layers
        """
        super().__init__()
        
        self.shape_before_flatten = shape_before_flatten
        C_flat, H_flat, W_flat = shape_before_flatten
        self.flat_dim = C_flat * H_flat * W_flat
        
        self.fc = nn.Linear(in_features=latent_dim, out_features=self.flat_dim)
        
        hidden_channels = hidden_channels[::-1]
        networks = []
        
        in_channels = hidden_channels[0]
        for i in range(len(hidden_channels) - 1):
            networks.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, 
                                   hidden_channels[i+1], 
                                   kernel_size=3, 
                                   stride=2, 
                                   padding=1, 
                                   output_padding=1),
                nn.BatchNorm2d(hidden_channels[i+1]),
                nn.ReLU()
            ))
            in_channels = hidden_channels[i+1]
            
        out_channels_final = output_shape[0]
        networks.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, 
                               out_channels=out_channels_final, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.Sigmoid()
        ))
        
        self.net = nn.Sequential(*networks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, *self.shape_before_flatten)
        x = self.net(x)
        return x

class ConvolutionalAE(Autoencoder):
    def __init__(self,
                 latent_dim: int,
                 input_shape: Tuple[int, int, int],
                 learn_rate: float = 1e-3,
                 number_of_epochs: int = 50,
                 hidden_channels: List[int] = [32, 64, 128]) -> None:
        super().__init__(
            latent_dim=latent_dim, 
            input_shape=input_shape, 
            learn_rate=learn_rate, 
            number_of_epochs=number_of_epochs
        )
        self.hidden_channels = hidden_channels

    def init_network(self) -> None:
        self.encoder = Encoder(self.input_shape, self.latent_dim, self.hidden_channels)
        
        shape_before_flatten = self.encoder.shape_before_flatten
        
        self.decoder = Decoder(self.latent_dim, self.input_shape, shape_before_flatten, self.hidden_channels)

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
        return "Convolutional Autoencoder"