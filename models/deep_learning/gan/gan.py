import torch
import torch.nn as nn
import torch.optim as optim
from ..gan import GAN

class Generator(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 img_shape: tuple) -> None:
        """
        Generator model of the GAN

        Parameters:
            latent_dim: Dimension of the noise vector
            img_shape: Shape of the output image
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=int(torch.prod(torch.tensor(self.img_shape)))),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator

        Parameters:
            z: Input noise tensor 

        Returns:
            generated_images: Tensor of generated images
        """
        img_flat = self.model(z)
        generated_images = img_flat.view(z.size(0), *self.img_shape)
        return generated_images

class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple) -> None:
        """
        Discriminator model of the GAN

        Parameters:
            img_shape: Shape of the input image 
        """
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=int(torch.prod(torch.tensor(self.img_shape))), out_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator

        Parameters:
            img: Input image tensor

        Returns:
            validity: Tensor representing real/fake probability
        """
        validity = self.model(img)
        return validity

class VanillaGAN(GAN):
    def init_network(self) -> None:
        """
        Initialize the generator, discriminator, optimizers, and loss function
        """
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate)
        self.criterion = nn.BCELoss()

    def __str__(self) -> str:
        return "Vanilla Generative Adversarial Network (GAN)"
