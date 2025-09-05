import torch
import torch.nn as nn
from ..cnn import ConvNet

class GhostModule(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 ratio: int = 2, 
                 kernel_size: int = 1, 
                 stride: int = 1, 
                 relu: bool = True) -> None:
        """
        Ghost Module: generate more feature maps from cheap operations

        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            ratio: Ratio to reduce channels for primary conv
            kernel_size: Kernel size for primary conv
            stride: Stride for primary conv
            relu: Whether to use ReLU activation
        """
        super().__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=init_channels, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=kernel_size // 2, 
                      bias=False),
            nn.BatchNorm2d(num_features=init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(in_channels=init_channels, 
                      out_channels=new_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      groups=init_channels, 
                      bias=False),
            nn.BatchNorm2d(num_features=new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GhostNet(ConvNet):
    def init_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=16, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            GhostModule(in_channels=16, out_channels=32, stride=1),
            nn.MaxPool2d(kernel_size=2),  

            GhostModule(in_channels=32, out_channels=64, stride=1),
            nn.MaxPool2d(kernel_size=2),  

            GhostModule(in_channels=64, out_channels=128, stride=1),
            GhostModule(in_channels=128, out_channels=256, stride=1),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=10)
        )
        self.network.apply(self.init_weights)

    def __str__(self):
        return "Convolutional Neural Networks: GhostNet"
