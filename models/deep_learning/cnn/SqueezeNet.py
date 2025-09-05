import torch
import torch.nn as nn
from ..cnn import ConvNet

class Fire(nn.Module):
    """
    Fire module from SqueezeNet: 1x1 squeeze convolution -> 1x1 expand convolution + 3x3 expand convolution
    """
    def __init__(self, 
                 in_channels: int, 
                 squeeze_channels: int, 
                 expand1x1_channels: int, 
                 expand3x3_channels: int) -> None:
        """
        Initialize the Fire module
        
        Parameters:
            in_channels: Number of input channels
            squeeze_channels: Number of squeeze channels
            expand1x1_channels: Number of 1x1 expand channels
            expand3x3_channels: Number of 3x3 expand channels
        """
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(num_features=squeeze_channels)
        self.squeeze_relu = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand1x1_channels, kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(num_features=expand1x1_channels)

        self.expand3x3 = nn.Conv2d(in_channels=squeeze_channels, 
                                   out_channels=expand3x3_channels, 
                                   kernel_size=3, 
                                   padding=1)
        self.expand3x3_bn = nn.BatchNorm2d(num_features=expand3x3_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fire module"""
        x = self.squeeze_relu(self.squeeze_bn(self.squeeze(x)))
        out1 = self.expand1x1_bn(self.expand1x1(x))
        out3 = self.expand3x3_bn(self.expand3x3(x))
        out = torch.cat([out1, out3], 1)
        return self.relu(out)


class SqueezeNet(ConvNet):
    def init_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=96, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),  
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            Fire(in_channels=96, 
                 squeeze_channels=16, 
                 expand1x1_channels=64, 
                 expand3x3_channels=64),  
            Fire(in_channels=128, 
                 squeeze_channels=16, 
                 expand1x1_channels=64, 
                 expand3x3_channels=64), 
            Fire(in_channels=128, 
                 squeeze_channels=32, 
                 expand1x1_channels=128, 
                 expand3x3_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Fire(in_channels=256, 
                 squeeze_channels=32, 
                 expand1x1_channels=128, 
                 expand3x3_channels=128),
            Fire(in_channels=256, 
                 squeeze_channels=48, 
                 expand1x1_channels=192, 
                 expand3x3_channels=192),
            Fire(in_channels=384, 
                 squeeze_channels=48, 
                 expand1x1_channels=192, 
                 expand3x3_channels=192),
            Fire(in_channels=384, 
                 squeeze_channels=64, 
                 expand1x1_channels=256, 
                 expand3x3_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            Fire(in_channels=512, 
                 squeeze_channels=64, 
                 expand1x1_channels=256, 
                 expand3x3_channels=256),

            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1),  
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.network.apply(self.init_weights)

    def __str__(self) -> str:
        return "Convolutional Neural Networks: SqueezeNet"
