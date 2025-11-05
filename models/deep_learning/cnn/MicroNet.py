import torch
import torch.nn as nn
from ..cnn import ConvNet

class MicroBlock(nn.Module):
    """
    MicroBlock = Depthwise Conv + Pointwise Conv + Non-linearity
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 2, 
                 expansion: int = 6) -> None:
        """
        Initialize the MicroBlock
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the depthwise convolution
            expansion: Expansion factor for the block
        """
        super().__init__()
        mid_channels = in_channels * expansion

        self.block = nn.Sequential(
            # 1x1 expansion
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=mid_channels, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU6(inplace=True),

            # 3x3 depthwise
            nn.Conv2d(in_channels=mid_channels, 
                      out_channels=mid_channels, 
                      kernel_size=3,
                      stride=stride, 
                      padding=1, 
                      groups=mid_channels, 
                      bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU6(inplace=True),

            # 1x1 projection
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.shortcut = (
            nn.Identity() if stride == 1 and in_channels == out_channels else None
        )
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.relu(out)


class MicroNet(ConvNet):
    def init_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            MicroBlock(in_channels=32,  out_channels=64),

            MicroBlock(in_channels=64,  out_channels=128),

            MicroBlock(in_channels=128,  out_channels=256),

            MicroBlock(in_channels=256, out_channels=512),

            MicroBlock(in_channels=512, 
                       out_channels=512, 
                       stride=1),

            nn.Conv2d(in_channels=512, 
                      out_channels=1024, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU6(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=self.out_channels)
        )

    def __str__(self):
        return "Convolutional Neural Networks: MicroNet"