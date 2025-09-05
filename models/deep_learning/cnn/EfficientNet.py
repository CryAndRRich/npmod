import torch
import torch.nn as nn
from ..cnn import ConvNet

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, 
                 in_channels: int, 
                 reduction: int = 4) -> None:
        """
        Initialize the SE Block
        
        Parameters:
            in_channels: Number of input channels
            reduction: Reduction ratio for the hidden layer
        """
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=hidden, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x)
        return x * scale


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution with SE"""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int, 
                 expand_ratio: int, 
                 reduction: int = 4) -> None:
        """
        Initialize the MBConv Block
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the depthwise convolution
            expand_ratio: Expansion ratio for the block
            reduction: Reduction ratio for the SE block
        """
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=hidden_dim, 
                          kernel_size=1, 
                          bias=False),
                nn.BatchNorm2d(num_features=hidden_dim),
                nn.SiLU(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(in_channels=hidden_dim, 
                      out_channels=hidden_dim, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1, 
                      groups=hidden_dim, 
                      bias=False),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.SiLU(inplace=True),

            SEBlock(in_channels=hidden_dim, reduction=reduction),

            nn.Conv2d(in_channels=hidden_dim, 
                      out_channels=out_channels, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            return x + out
        else:
            return out


class EfficientNet(ConvNet):
    def init_network(self):
        layers = []
        # Stem
        layers.extend([
            nn.Conv2d(in_channels=1, 
                      out_channels=16, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.SiLU(inplace=True)
        ])

        cfgs = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 2, 2),
            (6, 48, 2, 1),
            (6, 64, 2, 1),
            (6, 96, 2, 2),
            (6, 160, 1, 1),
        ]

        in_channels = 16
        for expand_ratio, out_channels, n, s in cfgs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(MBConv(in_channels=in_channels, 
                                     out_channels=out_channels, 
                                     stride=stride, 
                                     expand_ratio=expand_ratio))
                in_channels = out_channels

        # Head
        layers.extend([
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=320, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=320),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=320, out_features=10)
        ])

        self.network = nn.Sequential(*layers)
        self.network.apply(self.init_weights)

    def __str__(self):
        return "Convolutional Neural Networks: EfficientNet-Lite"
