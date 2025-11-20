import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from .StyleGAN import ToRGB, minibatch_stddev
from ..gan import GAN

class GenBlock(nn.Module):
    def __init__(self, 
                 in_c: int, 
                 out_c: int) -> None:
        """
        Generator block consisting of two convolutional layers followed by LeakyReLU activations
        
        Parameters:
            in_c: Number of input channels
            out_c: Number of output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, 
                               out_channels=out_c, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_c, 
                               out_channels=out_c, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        return x

class Generator(nn.Module):
    def __init__(self, 
                 latent_dim: int = 512, 
                 fmap_base: int = 512, 
                 max_resolution: int = 256) -> None:
        """
        Progressive Growing GAN Generator
        
        Parameters:
            latent_dim: Dimension of the latent vector
            fmap_base: Base number of feature maps
            max_resolution: Maximum output image resolution (must be a power of 2)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.max_log2 = int(math.log2(max_resolution))
        self.fmap_base = fmap_base
        self.num_layers = self.max_log2 - 2  # Starting from 4x4

        self.initial_block = GenBlock(fmap_base, fmap_base)
        self.fc = nn.Linear(in_features=latent_dim, out_features=fmap_base * 4 * 4)

        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList([ToRGB(in_channels=fmap_base)])

        in_c = fmap_base
        for i in range(1, self.num_layers + 1):
            out_c = max(fmap_base // (2 ** i), 32)
            self.blocks.append(GenBlock(in_c, out_c))
            self.to_rgbs.append(ToRGB(in_channels=out_c))
            in_c = out_c

    def forward(self, 
                z: torch.Tensor, 
                alpha: float = 1.0, 
                step: int = None) -> torch.Tensor:
        """
        Forward pass of the generator
        
        Parameters:
            z: Input latent vector
            alpha: Fade-in alpha for progressive growing
            step: Current step in progressive growing (None for max resolution)
        """
        B = z.size(0)
        if step is None:
            step = len(self.blocks)  

        x = self.fc(z).view(B, self.fmap_base, 4, 4)
        x = self.initial_block(x)

        rgb = self.to_rgbs[0](x)
        for i in range(step):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.blocks[i](x)
            rgb_new = self.to_rgbs[i + 1](x)
            if i == step - 1 and step > 0:
                rgb = F.interpolate(rgb, scale_factor=2, mode="nearest")
                rgb = rgb * (1 - alpha) + rgb_new * alpha
            else:
                rgb = rgb_new
        return torch.tanh(rgb)

class DiscBlock(nn.Module):
    def __init__(self, 
                 in_c: int, 
                 out_c: int) -> None:
        """
        Discriminator block consisting of two convolutional layers followed by LeakyReLU activations and downsampling
        
        Parameters:
            in_c: Number of input channels
            out_c: Number of output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, 
                               out_channels=out_c, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_c, 
                               out_channels=out_c, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.downsample(x)
        return x

class FromRGB(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, 
                              out_channels=out_channels, 
                              kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, 
                 fmap_base: int = 512, 
                 max_resolution: int = 256) -> None:
        """
        Progressive Growing GAN Discriminator
        
        Parameters:
            fmap_base: Base number of feature maps
            max_resolution: Maximum input image resolution (must be a power of 2)
        """
        super().__init__()
        self.max_log2 = int(math.log2(max_resolution))
        self.fmap_base = fmap_base
        self.num_layers = self.max_log2 - 2

        channels = [max(self.fmap_base // (2 ** i), 32) for i in range(self.num_layers + 1)]
        
        self.from_rgbs = nn.ModuleList([FromRGB(ch) for ch in channels])
        self.blocks = nn.ModuleList()
        
        for i in range(self.num_layers):
            in_c = channels[i + 1]
            out_c = channels[i]
            self.blocks.append(DiscBlock(in_c, out_c))

        final_in_c = channels[0]
        self.final_conv = nn.Conv2d(in_channels=final_in_c + 1, 
                                    out_channels=final_in_c, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1)
        self.final_fc = nn.Linear(in_features=final_in_c * 4 * 4, out_features=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, 
                img: torch.Tensor, 
                alpha: float = 1.0, 
                step: int = None) -> torch.Tensor:
        """
        Forward pass of the discriminator
        
        Parameters:
            img: Input image tensor
            alpha: Fade-in alpha for progressive growing
            step: Current step in progressive growing (None for max resolution)
        """
        if step is None: 
            step = len(self.blocks)
        
        out = self.from_rgbs[step](img)
        
        if step > 0 and alpha < 1.0:
            downsampled_img = F.avg_pool2d(img, 2)
            prev_out = self.from_rgbs[step - 1](downsampled_img)
            
            out = self.blocks[step - 1](out) 
            out = alpha * out + (1 - alpha) * prev_out
            start_index = step - 2
        else:
            start_index = step - 1

        for i in range(start_index, -1, -1):
            out = self.blocks[i](out)

        out = minibatch_stddev(out)
        out = self.lrelu(self.final_conv(out))
        out = out.view(out.size(0), -1)
        return self.final_fc(out)

class ProGAN(GAN):
    def __init__(self, 
                 latent_dim: int = 512, 
                 img_size: int = 256, 
                 fmap_base: int = 512, 
                 learn_rate: float = 1e-3, 
                 number_of_epochs: int = 100,
                 fade_in_epochs: int = 10,
                 n_critic: int = 1, 
                 r1_gamma: int = 10) -> None:
        """
        Progressive Growing GAN
        
        Parameters:
            latent_dim: Dimension of the latent vector
            img_size: Maximum output image resolution (must be a power of 2)
            fmap_base: Base number of feature maps
            learn_rate: Learning rate for optimizers
            number_of_epochs: Number of epochs to train at each resolution
            fade_in_epochs: Number of epochs to fade in new layers
            n_critic: Number of discriminator updates per generator update
            r1_gamma: Weight for R1 regularization
        """
        super().__init__(
            latent_dim=latent_dim,
            img_shape=None, 
            learn_rate=learn_rate, 
            number_of_epochs=number_of_epochs
        )
        self.img_size = img_size
        self.fmap_base = fmap_base
        self.fade_in_epochs = fade_in_epochs
        self.n_critic = n_critic
        self.r1_gamma = r1_gamma

    def init_network(self) -> None:
        self.generator = Generator(self.latent_dim, self.fmap_base, self.img_size)
        self.discriminator = Discriminator(self.fmap_base, self.img_size)
        
        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.0, 0.99))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.0, 0.99))
    
    def g_loss(self, fake):
        fake_logits = self.discriminator(fake)
        return F.softplus(-fake_logits).mean()
    
    @staticmethod
    def d_logistic_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
        return loss

    @staticmethod
    def g_nonsat_loss(fake_logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(-fake_logits).mean()

    @staticmethod
    def r1_regularization(real_generated_images: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        grads = autograd.grad(outputs=real_logits.sum(), inputs=real_generated_images, create_graph=True)[0]
        return grads.pow(2).reshape(grads.size(0), -1).sum(1).mean()

    def fit(self, 
            dataloader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        self.init_network()
        self.generator.train()
        self.discriminator.train()
        total_steps = len(self.generator.blocks) + 1

        for step in range(total_steps):
            curr_res = 4 * (2 ** step)
            num_batches = len(dataloader)
            for epoch in range(self.number_of_epochs):
                epoch_loss_G, epoch_loss_D = 0.0, 0.0
                batch_counter = 0

                for real_images, _ in dataloader:
                    batch_counter += 1
                    real_images = real_images * 2.0 - 1.0
                    if real_images.shape[-1] != curr_res:
                        real_images = F.interpolate(real_images, size=(curr_res, curr_res), mode="bilinear", align_corners=False)

                    total_batches = max(1, self.fade_in_epochs * num_batches)
                    current_batch = epoch * num_batches + batch_counter
                    alpha = min(1.0, current_batch / total_batches)
                    batch_size = real_images.size(0)

                    # Train Discriminator
                    for _ in range(self.n_critic):
                        z = torch.randn(batch_size, self.latent_dim)
                        fake_images = self.generator(z, alpha=alpha, step=step).detach()
                        real_logits = self.discriminator(real_images, alpha=alpha, step=step)
                        fake_logits = self.discriminator(fake_images, alpha=alpha, step=step)

                        d_loss = self.d_logistic_loss(real_logits, fake_logits)

                        self.optimizer_D.zero_grad()
                        d_loss.backward()
                        if self.r1_gamma is not None and self.r1_gamma > 0.0:
                            # Compute R1 on real images
                            real_images.requires_grad_(True)
                            real_logits_for_r1 = self.discriminator(real_images, alpha=alpha, step=step)
                            r1 = self.r1_regularization(real_images, real_logits_for_r1)
                            (0.5 * self.r1_gamma * r1).backward()
                            real_images.requires_grad_(False)
                        self.optimizer_D.step()

                    # Train Generator
                    z = torch.randn(batch_size, self.latent_dim)
                    fake_images = self.generator(z, alpha=alpha, step=step)
                    fake_logits = self.discriminator(fake_images, alpha=alpha, step=step)
                    g_loss = self.g_nonsat_loss(fake_logits)

                    self.optimizer_G.zero_grad()
                    g_loss.backward()
                    self.optimizer_G.step()

                    epoch_loss_D += float(d_loss.item())
                    epoch_loss_G += float(g_loss.item())

                epoch_loss_D /= num_batches
                epoch_loss_G /= num_batches

                if verbose and (epoch + 1) % 50 == 0:
                    print(f"Step [{step + 1}/{total_steps}] | Epoch [{epoch + 1}/{self.number_of_epochs}] | "
                          f"Loss_D: {epoch_loss_D:.4f} | Loss_G: {epoch_loss_G:.4f}")

    def generate(self, 
                 z: torch.Tensor, 
                 alpha: float = 1.0, 
                 step: int = None) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(z, alpha=alpha, step=step)
            generated_images = (generated_images.clamp(-1, 1) + 1) / 2.0
        return generated_images

    def __str__(self) -> str:
        return "Progressive Growing GAN (ProGAN)"