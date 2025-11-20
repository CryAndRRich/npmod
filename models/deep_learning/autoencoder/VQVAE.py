import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch.optim as optim
from ..autoencoder import Autoencoder

class VectorQuantizer(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 commitment_cost: float = 0.25) -> None:
        """
        Vector Quantization Layer for VQ-VAE
        
        Parameters:
            num_embeddings: Size of the dictionary (Number of vectors in codebook)
            embedding_dim: Dimension of each vector
            commitment_cost: Weight for commitment loss 
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self.embedding_dim)
        
        dist = (torch.sum(flat_input ** 2, dim=1, keepdim=True) 
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, quantized, encoding_indices
    
class Encoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 hidden_channels: List[int], 
                 latent_dim: int):
        super().__init__()
        
        C, _, _ = input_shape
        networks = []
        in_channels = C
        
        for h_dim in hidden_channels:
            networks.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=h_dim, 
                          kernel_size=4, 
                          stride=2, 
                          padding=1),
                nn.BatchNorm2d(num_features=h_dim),
                nn.ReLU()
            ))
            in_channels = h_dim
            
        networks.append(nn.Conv2d(in_channels=in_channels, 
                                 out_channels=latent_dim, 
                                 kernel_size=3, 
                                 stride=1, 
                                 padding=1))
        
        self.net = nn.Sequential(*networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 output_shape: Tuple[int, int, int], 
                 hidden_channels: List[int]):
        super().__init__()
        
        hidden_channels = hidden_channels[::-1]
        networks = []
        
        in_channels = latent_dim
        for i in range(len(hidden_channels)):
            networks.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, 
                                   out_channels=hidden_channels[i], 
                                   kernel_size=4, 
                                   stride=2, 
                                   padding=1),
                nn.BatchNorm2d(num_features=hidden_channels[i]),
                nn.ReLU()
            ))
            in_channels = hidden_channels[i]
            
        networks.append(nn.Conv2d(in_channels=in_channels, 
                                  out_channels=output_shape[0], 
                                  kernel_size=3, 
                                  padding=1))
        networks.append(nn.Sigmoid()) 
        
        self.net = nn.Sequential(*networks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
    

class VQVAE(Autoencoder):
    def __init__(self,
                 latent_dim: int,
                 input_shape: Tuple[int, int, int],
                 num_embeddings: int = 512,
                 learn_rate: float = 1e-3,
                 number_of_epochs: int = 50,
                 hidden_channels: List[int] = [32, 64],
                 commitment_cost: float = 0.25) -> None:
        """
        Vector Quantized Variational Autoencoder
        
        Parameters:
            latent_dim: Dimension of the latent space
            input_shape: Shape of the input data (C, H, W)
            num_embeddings: Number of vectors in the codebook
            learn_rate: Learning rate for optimizer
            number_of_epochs: Number of training epochs
            hidden_channels: List of hidden channel sizes for encoder/decoder
            commitment_cost: Weight for commitment loss
        """
        super().__init__(
            latent_dim=latent_dim, 
            input_shape=input_shape, 
            learn_rate=learn_rate, 
            number_of_epochs=number_of_epochs
        )
        self.num_embeddings = num_embeddings
        self.hidden_channels = hidden_channels
        self.commitment_cost = commitment_cost

    def init_network(self) -> None:
        self.encoder = Encoder(self.input_shape, self.hidden_channels, self.latent_dim)
        self.quantizer = VectorQuantizer(self.num_embeddings, self.latent_dim, self.commitment_cost)
        self.decoder = Decoder(self.latent_dim, self.input_shape, self.hidden_channels)

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        params = list(self.encoder.parameters()) + \
                 list(self.decoder.parameters()) + \
                 list(self.quantizer.parameters())
        self.optimizer = optim.Adam(params, lr=self.learn_rate)
        self.criterion = nn.MSELoss()

    def fit(self, 
            dataloader: torch.utils.data.DataLoader, 
            verbose: bool = False) -> None:
        self.init_network()
        self.encoder.train()
        self.quantizer.train()
        self.decoder.train()

        for epoch in range(self.number_of_epochs):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_vq_loss = 0.0
            
            for img, _ in dataloader:
                z_e = self.encoder(img) 
                
                vq_loss, z_q, _ = self.quantizer(z_e)
                
                reconstruction = self.decoder(z_q)
                
                recon_loss = self.criterion(reconstruction, img)
                loss = recon_loss + vq_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | "
                      f"Total Loss: {total_loss/len(dataloader):.4f} "
                      f"(Recon: {total_recon_loss/len(dataloader):.4f}, VQ: {total_vq_loss/len(dataloader):.4f})")
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        self.quantizer.eval()
        self.decoder.eval()
        with torch.no_grad():
            z_e = self.encoder(x)
            _, z_q, _ = self.quantizer(z_e)
            out = self.decoder(z_q)
        return out
    
    def __str__(self) -> str:
        return "Vector Quantized Variational Autoencoder (VQ-VAE)"