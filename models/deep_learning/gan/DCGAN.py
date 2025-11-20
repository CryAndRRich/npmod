import torch
import torch.nn as nn
import torch.optim as optim
from ..gan import GAN

class Generator(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 img_shape: int, 
                 feature_maps: int = 64) -> None:
        """
        Generator model of the DCGAN

        Parameters:
            latent_dim: Dimension of the noise vector
            img_shape: Number of channels of the output image 
            feature_maps: Number of feature maps 
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, 
                               out_channels=feature_maps * 8, 
                               kernel_size=4, 
                               stride=1, 
                               padding=0, 
                               bias=False),
            nn.BatchNorm2d(num_features=feature_maps * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=feature_maps * 8, 
                               out_channels=feature_maps * 4, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(num_features=feature_maps * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=feature_maps * 4, 
                               out_channels=feature_maps * 2, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(num_features=feature_maps * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=feature_maps * 2, 
                               out_channels=feature_maps, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(num_features=feature_maps),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=feature_maps, 
                               out_channels=img_shape, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
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
        generated_images = self.model(z)
        return generated_images

class Discriminator(nn.Module):
    def __init__(self, 
                 img_shape: int, 
                 feature_maps: int = 64) -> None:
        """
        Discriminator model of the DCGAN

        Parameters:
            img_shape: Number of channels of the input image
            feature_maps: Number of feature maps 
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=img_shape, 
                      out_channels=feature_maps, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=feature_maps, 
                      out_channels=feature_maps * 2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=feature_maps * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=feature_maps * 2, 
                      out_channels=feature_maps * 4, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=feature_maps * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=feature_maps * 4, 
                      out_channels=feature_maps * 8, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=feature_maps * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=feature_maps * 8, 
                      out_channels=1, 
                      kernel_size=4, 
                      stride=1, 
                      padding=0, 
                      bias=False)
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
        return validity.view(-1, 1)

class DCGAN(GAN):
    def init_network(self) -> None:
        img_channels = self.img_shape[0]
        self.generator = Generator(self.latent_dim, img_channels)
        self.discriminator = Discriminator(img_channels)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, 
            dataloader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        self.init_network()
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.number_of_epochs):
            epoch_loss_G, epoch_loss_D = 0.0, 0.0

            for real_images, _ in dataloader:
                real_images = real_images
                batch_size = real_images.size(0)

                real_label = 0.9
                eps = 0.05

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_validity = self.discriminator(real_images)
                valid = torch.empty_like(real_validity).uniform_(real_label - eps, real_label + eps)
                loss_real = self.criterion(real_validity, valid)

                z = torch.randn(batch_size, self.latent_dim, 1, 1)
                generated_images = self.generator(z)
                fake_validity = self.discriminator(generated_images.detach())
                fake = torch.empty_like(fake_validity).uniform_(0.0, eps)
                loss_fake = self.criterion(fake_validity, fake)

                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                gen_validity = self.discriminator(generated_images)
                loss_G = self.criterion(gen_validity, valid)
                loss_G.backward()
                self.optimizer_G.step()

                epoch_loss_D += loss_D.item()
                epoch_loss_G += loss_G.item()
            
            epoch_loss_D /= len(dataloader)
            epoch_loss_G /= len(dataloader)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | "
                      f"Loss_D: {epoch_loss_D:.4f} | Loss_G: {epoch_loss_G:.4f}")

    def __str__(self) -> str:
        return "Deep Convolutional GAN (DCGAN)"