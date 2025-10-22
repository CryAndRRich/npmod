# This script tests various custom autoencoder models
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Importing the custom models
from models.deep_learning.autoencoder.dae import DenoisingAutoencoder
from models.deep_learning.autoencoder.vae import VariationalAutoencoder
from models.deep_learning.autoencoder.cae import ContractiveAutoencoder
from models.deep_learning.autoencoder.convae import ConvAutoencoder


# === Trainer for Autoencoders === #
class Trainer():
    def __init__(self, 
                 model: nn.Module, 
                 lr: float = 1e-3, 
                 epochs: int = 20, 
                 device: str = "cpu") -> None:
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device).float().view(batch.size(0), -1)

                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.model.loss_function(batch, output)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device).float().view(batch.size(0), -1)
                output = self.model(batch)
                loss = self.model.loss_function(batch, output)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

    def reconstruct(self, x: torch.Tensor) -> tuple:
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device).float().view(x.size(0), -1)
            output = self.model(x)
            if isinstance(output, tuple):  # VAE output
                output = output[0]
        return x, output
# ====================


if __name__ == "__main__":
    # === Load Dataset === 
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    # ====================


    # === Test Autoencoder === #
    models = [
        DenoisingAutoencoder(input_dim=784, latent_dim=64),
        VariationalAutoencoder(input_dim=784, latent_dim=20),
        ContractiveAutoencoder(input_dim=784, latent_dim=64),
        ConvAutoencoder()
    ]

    for model in models:
        trainer = Trainer(model, 
                        lr=1e-3, 
                        epochs=5, 
                        device="cuda" if torch.cuda.is_available() else "cpu")

        print("==============================================================")
        print(f"{model.__str__()} Results")
        print("==============================================================")
        trainer.fit(train_loader)
        print("==============================================================")
        trainer.evaluate(test_loader)

    """
    ==============================================================
    Denoising Autoencoder Results
    ==============================================================
    Epoch 1/5, Loss: 0.0506
    Epoch 2/5, Loss: 0.0228
    Epoch 3/5, Loss: 0.0172
    Epoch 4/5, Loss: 0.0144
    Epoch 5/5, Loss: 0.0125
    ==============================================================
    Evaluation Loss: 0.0114

    ==============================================================
    Variational Autoencoder Results
    ==============================================================
    Epoch 1/5, Loss: 0.2961
    Epoch 2/5, Loss: 0.2646
    Epoch 3/5, Loss: 0.2642
    Epoch 4/5, Loss: 0.2639
    Epoch 5/5, Loss: 0.2637
    ==============================================================
    Evaluation Loss: 0.2634

    ==============================================================
    Contractive Autoencoder Results
    ==============================================================
    Epoch 1/5, Loss: 0.0701
    Epoch 2/5, Loss: 0.0572
    Epoch 3/5, Loss: 0.0500
    Epoch 4/5, Loss: 0.0412
    Epoch 5/5, Loss: 0.0387
    ==============================================================
    Evaluation Loss: 0.0374

    ==============================================================
    Convolutional Autoencoder Results
    ==============================================================
    Epoch 1/5, Loss: 0.0222
    Epoch 2/5, Loss: 0.0018
    Epoch 3/5, Loss: 0.0012
    Epoch 4/5, Loss: 0.0009
    Epoch 5/5, Loss: 0.0008
    ==============================================================
    Evaluation Loss: 0.0007
    """