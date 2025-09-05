from typing import Tuple
import torch
import torch.nn as nn
from ..cnn import ConvNet

class Inception(nn.Module):
    """
    Inception Block for GoogLeNet

    This module applies multiple parallel transformations on the input and concatenates the outputs
    """
    def __init__(self,
                 in_channels: int,
                 c1: int,
                 c2: Tuple[int, int],
                 c3: Tuple[int, int],
                 c4: int,
                 **kwargs) -> None:
        """
        Parameters:
            in_channels: Number of input channels
            c1: Number of output channels for the 1x1 convolution in Path 1
            c2: Tuple (channels_1x1, channels_3x3) for Path 2
            c3: Tuple (channels_1x1, channels_5x5) for Path 3
            c4: Number of output channels for the 1x1 convolution in Path 4
        """
        super().__init__(**kwargs)

        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1),
            nn.BatchNorm2d(num_features=c1),
            nn.ReLU(inplace=True)
        )

        # Path 2: 1x1 convolution followed by a 3x3 convolution
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1),
            nn.BatchNorm2d(num_features=c2[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=c2[1]),
            nn.ReLU(inplace=True)
        )

        # Path 3: 1x1 convolution followed by a 5x5 convolution
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1),
            nn.BatchNorm2d(num_features=c3[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=c3[1]),
            nn.ReLU(inplace=True)
        )

        # Path 4: 3x3 max pooling followed by a 1x1 convolution
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1),
            nn.BatchNorm2d(num_features=c4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Inception module
        
        Parameters:
            x: Input tensor
            
        Returns:
            torch.Tensor: Concatenated output from all paths
        """
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        p4 = self.path4(x)
        return torch.cat(tensors=(p1, p2, p3, p4), dim=1)
    
class GoogLeNet(ConvNet):
    """
    GoogLeNet built using convolutional layers and Inception blocks,
    followed by an Adaptive Average Pooling layer and a fully-connected layer
    """
    def init_network(self):
        """
        Initializes the GoogLeNet network architecture
        """
        block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        block3 = nn.Sequential(
            Inception(in_channels=64, c1=16, c2=(16, 24), c3=(4, 8), c4=8),
            Inception(in_channels=56, c1=24, c2=(24, 32), c3=(8, 16), c4=16),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        block4 = nn.Sequential(
            Inception(in_channels=88, c1=32, c2=(32, 48), c3=(8, 16), c4=16)
        )

        self.network = nn.Sequential(
            block1,
            block2,
            block3,
            block4,

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=112, out_features=10)
        )
        self.network.apply(self.init_weights)
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: GoogLeNet (Inception)"
