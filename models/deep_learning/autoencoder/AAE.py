import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from ..autoencoder import Autoencoder

class Encoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 latent_dim: int, 
                 hidden_dims: List[int] = [512, 256]) -> None:
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
        networks.append(nn.Linear(in_features=in_dim, out_features=latent_dim))
        self.net = nn.Sequential(*networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 output_shape: Tuple[int, int, int], 
                 hidden_dims: List[int] = [512, 256]) -> None:
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

class Discriminator(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 hidden_dim: int = 256) -> None:
        """
        Discriminator Network for Adversarial Autoencoder

        Parameters:
            latent_dim: Dimension of the latent space
            hidden_dim: Dimension of the hidden layer
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=hidden_dim // 2, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class AAE(Autoencoder):
    def __init__(self,
                 latent_dim: int,
                 input_shape: Tuple[int, int, int],
                 learn_rate: float = 1e-3,
                 number_of_epochs: int = 50,
                 hidden_dims: List[int] = [512, 256]) -> None:
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
        
        self.discriminator = Discriminator(self.latent_dim)

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.opt_recon = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learn_rate)
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate * 0.5)
        self.opt_gen = optim.Adam(self.encoder.parameters(), lr=self.learn_rate)

        self.criterion_mse = nn.MSELoss()
        self.criterion_bce = nn.BCELoss()

    def fit(self, 
            dataloader: torch.utils.data.DataLoader, 
            verbose: bool = False) -> None:
        self.init_network()
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        for epoch in range(self.number_of_epochs):
            total_recon_loss = 0.0
            total_disc_loss = 0.0
            total_gen_loss = 0.0
            
            for img, _ in dataloader:
                batch_size = img.size(0)

                self.opt_recon.zero_grad()
                
                z = self.encoder(img)
                recon = self.decoder(z)
                
                recon_loss = self.criterion_mse(recon, img)
                recon_loss.backward()
                self.opt_recon.step()
                
                self.opt_disc.zero_grad()
                
                z_real = torch.randn(batch_size, self.latent_dim)
                z_fake = self.encoder(img).detach()
                
                pred_real = self.discriminator(z_real)
                pred_fake = self.discriminator(z_fake)
                
                loss_real = self.criterion_bce(pred_real, torch.ones_like(pred_real))
                loss_fake = self.criterion_bce(pred_fake, torch.zeros_like(pred_fake))
                
                disc_loss = 0.5 * (loss_real + loss_fake)
                disc_loss.backward()
                self.opt_disc.step()
                
                self.opt_gen.zero_grad()
                
                z_fake_gen = self.encoder(img)
                pred_fake_gen = self.discriminator(z_fake_gen)
                
                gen_loss = self.criterion_bce(pred_fake_gen, torch.ones_like(pred_fake_gen))
                gen_loss.backward()
                self.opt_gen.step()
                
                total_recon_loss += recon_loss.item()
                total_disc_loss += disc_loss.item()
                total_gen_loss += gen_loss.item()

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | "
                      f"Recon: {total_recon_loss/len(dataloader):.4f} | "
                      f"Disc: {total_disc_loss/len(dataloader):.4f} | "
                      f"Gen: {total_gen_loss/len(dataloader):.4f}")

    def __str__(self):
        return "Adversarial Autoencoder (AAE)"