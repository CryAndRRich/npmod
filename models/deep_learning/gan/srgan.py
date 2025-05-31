import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..gan import GAN

class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        """
        Residual block for SRGAN generator

        Parameters:
            channels: Number of feature maps in the block
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one residual block

        Parameters:
            x: Input feature map

        Returns:
            out: Output feature map after residual connection
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual

class UpsampleBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 up_scale: int) -> None:
        """
        Upsample block using pixel shuffle

        Parameters:
            in_channels: Number of input feature maps
            up_scale: Upsampling factor 
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=in_channels * (up_scale ** 2), 
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through upsample block

        Parameters:
            x: Input feature map

        Returns:
            out: Upsampled feature map
        """
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out

class Generator(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 num_residual_blocks: int = 16, 
                 up_scale: int = 4) -> None:
        """
        Generator (SRResNet) for SRGAN

        Parameters:
            in_channels: Number of input image channels 
            num_residual_blocks: Number of residual blocks to use
            up_scale: Total upsampling factor 
        """
        super().__init__()
        # First convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=64, 
                               kernel_size=9, 
                               stride=1, 
                               padding=4)
        self.prelu1 = nn.PReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels=64) for _ in range(num_residual_blocks)]
        )

        # Second conv after residual blocks
        self.conv2 = nn.Conv2d(in_channels=64, 
                               out_channels=64, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        # Upsampling layers (two stages of ×2 if up_scale=4)
        if up_scale == 2:
            self.upsample = nn.Sequential(UpsampleBlock(64, 2))
        elif up_scale == 4:
            self.upsample = nn.Sequential(UpsampleBlock(64, 2), UpsampleBlock(64, 2))
        else:
            raise ValueError("Unsupported up_scale, must be 2 or 4")

        # Final output conv
        self.conv3 = nn.Conv2d(in_channels=64, 
                               out_channels=in_channels, 
                               kernel_size=9, 
                               stride=1, 
                               padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SRGAN generator

        Parameters:
            x: Low-resolution input image tensor

        Returns:
            out: Super-resolved image tensor
        """
        out1 = self.prelu1(self.conv1(x))
        residual = out1
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.upsample(out)
        out = self.conv3(out)
        return torch.tanh(out)

class Discriminator(nn.Module):
    def __init__(self, in_channels: int) -> None:
        """
        Discriminator for SRGAN

        Parameters:
            in_channels: Number of channels of the high-resolution image 
        """
        super().__init__()
        def disc_block(in_c: int, 
                       out_c: int, 
                       stride: int, 
                       batch_norm: bool) -> nn.Sequential:
            layers = [nn.Conv2d(in_channels=in_c, 
                                out_channels=out_c, 
                                kernel_size=3, 
                                stride=stride, 
                                padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_features=out_c))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(in_channels, 64, stride=1, batch_norm=False),
            disc_block(64, 64, stride=2, batch_norm=True),

            disc_block(64, 128, stride=1, batch_norm=True),
            disc_block(128, 128, stride=2, batch_norm=True),

            disc_block(128, 256, stride=1, batch_norm=True),
            disc_block(256, 256, stride=2, batch_norm=True),

            disc_block(256, 512, stride=1, batch_norm=True),
            disc_block(512, 512, stride=2, batch_norm=True),

            nn.Flatten(),
            nn.Linear(in_features=512 * (16 * 16), out_features=1024),  
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SRGAN discriminator

        Parameters:
            x: High-resolution image tensor (generated or real)

        Returns:
            validity: Real/fake probability
        """
        return self.model(x)

class SRGAN(GAN):
    def __init__(self,
                 lr_shape: tuple,
                 hr_shape: tuple,
                 learn_rate: float,
                 number_of_epochs: int,
                 num_residual_blocks: int = 16,
                 up_scale: int = 4) -> None:
        """
        Super-Resolution GAN (SRGAN)

        Parameters:
            lr_shape: Shape of low-resolution input images (C, h, w)
            hr_shape: Shape of high-resolution target images (C, H, W)
            learn_rate: Learning rate for optimizers
            number_of_epochs: Number of training epochs
            num_residual_blocks: Number of residual blocks in generator
            up_scale: Upscaling factor (e.g., 4 for ×4 SR)
        """
        super().__init__(latent_dim=None, img_shape=None, learn_rate=learn_rate, number_of_epochs=number_of_epochs)
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.num_residual_blocks = num_residual_blocks
        self.up_scale = up_scale

    def init_network(self) -> None:
        """
        Initialize generator, discriminator, optimizers, and loss functions
        """
        C_lr, _, _ = self.lr_shape
        C_hr, _, _ = self.hr_shape

        self.generator = Generator(in_channels=C_lr,
                                   num_residual_blocks=self.num_residual_blocks,
                                   up_scale=self.up_scale)
        self.discriminator = Discriminator(in_channels=C_hr)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.9, 0.999))

        self.adversarial_criterion = nn.BCELoss()
        self.content_criterion = nn.MSELoss()

    def fit(self, dataloader: DataLoader) -> None:
        """
        Train the SRGAN model using paired low-res and high-res images

        Parameters:
            dataloader: DataLoader yielding tuples (lr_images, hr_images)
        """
        self.init_network()

        for _ in range(self.number_of_epochs):
            for lr_images, hr_images in dataloader:
                batch_size = lr_images.size(0)
                # Adversarial ground truths
                valid = torch.ones(batch_size, 1)
                fake = torch.zeros(batch_size, 1)

                # Generate high-resolution images from low-resolution inputs
                gen_hr = self.generator(lr_images)

                # Train Discriminator
                self.optimizer_D.zero_grad()

                real_validity = self.discriminator(hr_images)
                fake_validity = self.discriminator(gen_hr.detach())

                loss_D_real = self.adversarial_criterion(real_validity, valid)
                loss_D_fake = self.adversarial_criterion(fake_validity, fake)
                loss_D = (loss_D_real + loss_D_fake) / 2

                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()

                # Content loss (MSE between generated and real HR)
                content_loss = self.content_criterion(gen_hr, hr_images)

                # Adversarial loss (on discriminator's prediction of generated images)
                pred_fake = self.discriminator(gen_hr)
                adversarial_loss = self.adversarial_criterion(pred_fake, valid)

                # Total generator loss: content + 1e-3 * adversarial
                loss_G = content_loss + 1e-3 * adversarial_loss

                loss_G.backward()
                self.optimizer_G.step()

    def __str__(self) -> str:
        return "Super-Resolution GAN (SRGAN)"
