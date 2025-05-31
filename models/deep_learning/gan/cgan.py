from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .gan import GAN

class Generator(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 num_classes: int, 
                 img_shape: Tuple[int]) -> None:
        """
        Generator model for the CGAN

        Parameters:
            latent_dim: Dimension of the noise vector
            num_classes: Number of conditional classes
            img_shape: Shape of the generated images (C, H, W)
        """
        super().__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(norm_type=num_classes, embedding_dim=num_classes)

        input_dim = latent_dim + num_classes
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, 
                z: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator

        Parameters:
            z: Input noise tensor
            labels: Class labels as conditions

        Returns:
            generated_images: Tensor of generated images
        """
        label_input = self.label_emb(labels)
        gen_input = torch.cat((z, label_input), dim=1)
        img_flat = self.model(gen_input)
        generated_images = img_flat.view(img_flat.size(0), *self.img_shape)
        return generated_images


class Discriminator(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 img_shape: Tuple[int]) -> None:
        """
        Discriminator model for the CGAN

        Parameters:
            num_classes: Number of conditional classes
            img_shape: Shape of the input images (C, H, W)
        """
        super().__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)

        input_dim = num_classes + int(torch.prod(torch.tensor(img_shape)))
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, 
                img: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator

        Parameters:
            img: Input image tensor
            labels: Class labels as conditions

        Returns:
            validity: Tensor representing real/fake probability
        """
        label_input = self.label_emb(labels)
        img_flat = img.view(img.size(0), -1)
        d_in = torch.cat((img_flat, label_input), dim=1)
        validity = self.model(d_in)
        return validity

class CGAN(GAN):
    def __init__(self,
                 latent_dim: int,
                 img_shape: tuple,
                 learn_rate: float,
                 number_of_epochs: int,
                 num_classes: int) -> None:
        """
        Conditional GAN (CGAN)

        Parameters:
            latent_dim: Dimension of the noise vector
            img_shape: Shape of the generated images (C, H, W)
            learn_rate: Learning rate
            number_of_epochs: Number of training iterations
            num_classes: Number of conditional classes
            device: Device to train on
        """
        super().__init__(latent_dim, img_shape, learn_rate, number_of_epochs)
        self.num_classes = num_classes

    def init_network(self) -> None:
        """
        Initialize the generator, discriminator, optimizers, and loss function
        """
        self.generator = Generator(self.latent_dim, self.num_classes, self.img_shape)
        self.discriminator = Discriminator(self.num_classes, self.img_shape)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def fit(self, dataloader: DataLoader) -> None:
        """
        Train the CGAN using the provided DataLoader

        Parameters:
            dataloader: DataLoader for real images and labels
        """
        self.init_network()

        for _ in range(self.number_of_epochs):
            for real_images, labels in dataloader:
                real_images = real_images
                labels = labels
                batch_size = real_images.size(0)

                valid = torch.ones((batch_size, 1))
                fake = torch.zeros((batch_size, 1))

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_validity = self.discriminator(real_images, labels)
                loss_real = self.criterion(real_validity, valid)

                z = torch.randn(batch_size, self.latent_dim)
                gen_labels = torch.randint(0, self.num_classes, (batch_size,))
                generated_images = self.generator(z, gen_labels)
                fake_validity = self.discriminator(generated_images.detach(), gen_labels)
                loss_fake = self.criterion(fake_validity, fake)

                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                gen_validity = self.discriminator(generated_images, gen_labels)
                loss_G = self.criterion(gen_validity, valid)
                loss_G.backward()
                self.optimizer_G.step()

    def generate(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate images from noise vectors and labels using the trained generator

        Parameters:
            noise: Input noise tensor
            labels: Class labels as conditions

        Returns:
            generated_images: Tensor of generated images
        """
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(noise, labels)
        return generated_images

    def __str__(self) -> str:
        return "Conditional GAN (CGAN)"
