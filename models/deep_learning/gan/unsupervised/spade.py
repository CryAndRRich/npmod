import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.deep_learning.gan import GAN

class SPADE(nn.Module):
    def __init__(self, 
                 norm_nc: int, 
                 label_nc: int, 
                 ks: int = 3) -> None:
        """
        SPADE normalization layer

        Parameters:
            norm_nc: Number of channels in the normalized feature map
            label_nc: Number of channels in the semantic segmentation map
            ks: Kernel size for the shared convolution
        """
        super().__init__()
        padding = ks // 2
        # Shared convolution to process segmentation map
        self.shared_conv = nn.Conv2d(in_channels=label_nc, 
                                     out_channels=128, 
                                     kernel_size=ks, 
                                     stride=1, 
                                     padding=padding)
        self.bn_shared = nn.BatchNorm2d(num_features=128)
        self.relu_shared = nn.ReLU(inplace=True)
        # Separate conv layers to output gamma and beta
        self.gamma_conv = nn.Conv2d(in_channels=128, 
                                    out_channels=norm_nc, 
                                    kernel_size=ks, 
                                    stride=1, 
                                    padding=padding)
        self.beta_conv = nn.Conv2d(in_channels=128, 
                                   out_channels=norm_nc, 
                                   kernel_size=ks, 
                                   stride=1, 
                                   padding=padding)

        # Parameter-free normalization (instance norm)
        self.norm = nn.InstanceNorm2d(num_features=norm_nc, affine=False)

    def forward(self, 
                x: torch.Tensor, 
                segmap: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SPADE

        Parameters:
            x: Input feature map (B, norm_nc, H, W)
            segmap: Semantic segmentation map (B, label_nc, H, W)

        Returns:
            out: Normalized and modulated feature map
        """
        seg_feat = self.shared_conv(segmap)
        seg_feat = self.bn_shared(seg_feat)
        seg_feat = self.relu_shared(seg_feat)
        gamma = self.gamma_conv(seg_feat)
        beta = self.beta_conv(seg_feat)
        normalized = self.norm(x)
        return normalized * (1 + gamma) + beta

class ResnetBlock(nn.Module):
    def __init__(self, 
                 fin: int, 
                 fout: int, 
                 label_nc: int) -> None:
        """
        ResNet block using SPADE layers

        Parameters:
            fin: Number of input channels
            fout: Number of output channels
            label_nc: Number of channels in segmentation map
        """
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # SPADE + conv1
        self.spade1 = SPADE(fin, label_nc)
        self.conv1 = nn.Conv2d(in_channels=fin, 
                               out_channels=fmiddle, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=fmiddle)

        # SPADE + conv2
        self.spade2 = SPADE(fmiddle, label_nc)
        self.conv2 = nn.Conv2d(in_channels=fmiddle, 
                               out_channels=fout, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=fout)

        # If needed, a conv for the skip connection
        if self.learned_shortcut:
            self.spade_sc = SPADE(fin, label_nc)
            self.conv_sc = nn.Conv2d(in_channels=fin, 
                                     out_channels=fout, 
                                     kernel_size=1, 
                                     stride=1, 
                                     padding=0)

    def forward(self, 
                x: torch.Tensor, 
                segmap: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SPADE ResNet block

        Parameters:
            x: Input feature map
            segmap: Semantic segmentation map
        Returns:
            out: Output feature map after SPADE ResNet block
        """
        # Shortcut branch
        if self.learned_shortcut:
            x_s = self.spade_sc(x, segmap)
            x_s = self.conv_sc(x_s)
        else:
            x_s = x

        # Main branch
        dx = self.spade1(x, segmap)
        dx = self.conv1(dx)
        dx = self.bn1(dx)
        dx = nn.ReLU(inplace=True)(dx)

        dx = self.spade2(dx, segmap)
        dx = self.conv2(dx)
        dx = self.bn2(dx)

        return x_s + dx

class Generator(nn.Module):
    def __init__(self, 
                 label_nc: int, 
                 output_nc: int = 3, 
                 ngf: int = 64, 
                 num_res_blocks: int = 6) -> None:
        """
        SPADE Generator network

        Parameters:
            label_nc: Number of semantic segmentation map channels
            output_nc: Number of output image channels 
            ngf: Base number of filters
            num_res_blocks: Number of SPADE ResNet blocks
        """
        super().__init__()
        self.fc = nn.Conv2d(in_channels=label_nc, 
                            out_channels=16 * ngf, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1)
        self.res_blocks = nn.ModuleList()
        nf = 16 * ngf
        for _ in range(num_res_blocks):
            block = ResnetBlock(nf, nf, label_nc)
            self.res_blocks.append(block)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up1 = nn.Conv2d(in_channels=nf, 
                                  out_channels=8 * ngf, 
                                  kernel_size=3, 
                                  stride=1, 
                                  padding=1)
        self.res_block1 = ResnetBlock(8 * ngf, 8 * ngf, label_nc)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up2 = nn.Conv2d(in_channels=8 * ngf, 
                                  out_channels=4 * ngf, 
                                  kernel_size=3, 
                                  stride=1, 
                                  padding=1)
        self.res_block2 = ResnetBlock(4 * ngf, 4 * ngf, label_nc)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up3 = nn.Conv2d(in_channels=4 * ngf, 
                                  out_channels=2 * ngf, 
                                  kernel_size=3, 
                                  stride=1, 
                                  padding=1)
        self.res_block3 = ResnetBlock(2 * ngf, 2 * ngf, label_nc)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up4 = nn.Conv2d(in_channels=2 * ngf, 
                                  out_channels=ngf, 
                                  kernel_size=3, 
                                  stride=1, 
                                  padding=1)
        self.res_block4 = ResnetBlock(ngf, ngf, label_nc)

        self.final_conv = nn.Conv2d(in_channels=ngf, 
                                    out_channels=output_nc, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1)
        self.tanh = nn.Tanh()

    def forward(self, segmap: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SPADE Generator

        Parameters:
            segmap: Semantic segmentation map tensor
        Returns:
            Generated image tensor
        """
        x = self.fc(segmap)
        for block in self.res_blocks:
            x = block(x, segmap)

        x = self.upsample1(x)
        x = self.conv_up1(x)
        x = self.res_block1(x, segmap)

        x = self.upsample2(x)
        x = self.conv_up2(x)
        x = self.res_block2(x, segmap)

        x = self.upsample3(x)
        x = self.conv_up3(x)
        x = self.res_block3(x, segmap)

        x = self.upsample4(x)
        x = self.conv_up4(x)
        x = self.res_block4(x, segmap)

        x = self.final_conv(x)
        return self.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, 
                 input_nc: int = 3, 
                 ndf: int = 64, 
                 num_layers: int = 3) -> None:
        """
        PatchGAN Discriminator for SPADE

        Parameters:
            input_nc: Number of channels of input image
            ndf: Base number of filters
            num_layers: Number of convolutional layers
        """
        super().__init__()
        sequence = [
            nn.Conv2d(in_channels=input_nc + 1, 
                      out_channels=ndf, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(in_channels=ndf * nf_mult_prev, 
                          out_channels=ndf * nf_mult, 
                          kernel_size=4, 
                          stride=2, 
                          padding=1),
                nn.BatchNorm2d(num_features=ndf * nf_mult),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
        # Final conv
        sequence += [nn.Conv2d(in_channels=ndf * nf_mult, 
                               out_channels=1, 
                               kernel_size=4, 
                               stride=1, 
                               padding=1)]
        sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, 
                img: torch.Tensor, 
                segmap: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SPADE discriminator

        Parameters:
            img: Real or generated image tensor
            segmap: Semantic segmentation map tensor (resized to img size)

        Returns:
            Patch-wise real/fake probability map
        """
        x = torch.cat((img, segmap), 1)
        return self.model(x)

class SPADE(GAN):
    def __init__(self,
                 label_nc: int,
                 image_nc: int,
                 learn_rate: float,
                 number_of_epochs: int,
                 lambda_lp: float = 10.0) -> None:
        """
        SPADE-based Semantic Image Synthesis GAN

        Parameters:
            label_nc: Number of channels in segmentation map
            image_nc: Number of channels in output image
            learn_rate: Learning rate for optimizers
            number_of_epochs: Number of training epochs
            lambda_lp: Weight for feature matching or perceptual loss (not implemented)
        """
        super().__init__(latent_dim=None, img_shape=(image_nc,), learn_rate=learn_rate, number_of_epochs=number_of_epochs)
        self.label_nc = label_nc
        self.image_nc = image_nc
        self.lambda_lp = lambda_lp

    def init_network(self) -> None:
        """
        Initialize SPADE generator, PatchGAN discriminator, optimizers, and losses
        """
        self.generator = Generator(label_nc=self.label_nc, output_nc=self.image_nc)
        self.discriminator = Discriminator(input_nc=self.image_nc)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))

        self.criterion_GAN = nn.BCELoss()

    def fit(self, dataloader: DataLoader) -> None:
        """
        Train SPADE with (segmap, real_image) pairs

        Parameters:
            dataloader: DataLoader yielding (segmap, real_image) pairs
        """
        self.init_network()
        for _ in range(self.number_of_epochs):
            for segmap, real_img in dataloader:
                batch_size = segmap.size(0)
                valid = torch.ones((batch_size, 1, 30, 30))
                fake = torch.zeros((batch_size, 1, 30, 30))

                # Train Discriminator
                self.optimizer_D.zero_grad()
                gen_img = self.generator(segmap)
                pred_real = self.discriminator(real_img, segmap)
                pred_fake = self.discriminator(gen_img.detach(), segmap)
                loss_D_real = self.criterion_GAN(pred_real, valid)
                loss_D_fake = self.criterion_GAN(pred_fake, fake)
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                pred_fake_for_G = self.discriminator(gen_img, segmap)
                loss_G_GAN = self.criterion_GAN(pred_fake_for_G, valid)
                # Additional perceptual losses could be added
                loss_G = loss_G_GAN
                loss_G.backward()
                self.optimizer_G.step()

    def __str__(self) -> str:
        return "Spatially-Adaptive Normalization (SPADE)"
