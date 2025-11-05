import torch
import torch.nn as nn
from ..cnn import ConvNet

class Residual(nn.Module):
    """
    Residual Block for ResNet

    A residual block consists of two convolutional layers with batch normalization and ReLU activation.
    It includes a skip connection that either passes the input directly or transforms it using a 1x1 convolution
    """
    def __init__(self,
                 input_channels: int, 
                 num_channels: int, 
                 use_1x1_conv: bool = False, 
                 strides: int = 1) -> None:
        """
        Parameters:
            input_channels: Number of input channels
            num_channels: Number of output channels
            use_1x1_conv: If True, applies a 1x1 convolution to the input in the skip connection to match the shape
            strides: Stride value for the first convolutional layer
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=num_channels, 
                      kernel_size=3, 
                      padding=1, 
                      stride=strides,
                      bias=False),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=num_channels, 
                      out_channels=num_channels, 
                      kernel_size=3, 
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=num_channels)
        )

        self.shortcut = None
        if use_1x1_conv:
            self.shortcut = nn.Conv2d(in_channels=input_channels, 
                                      out_channels=num_channels, 
                                      kernel_size=1, 
                                      stride=strides,
                                      bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.shortcut:
            x = self.shortcut(x)
        return self.relu(y + x)

def resnet_block(input_channels: int, 
                 num_channels: int, 
                 num_residuals: int,
                 first_block: bool = False) -> nn.Sequential:
    """
    Creates a ResNet block consisting of multiple residual units

    Parameters:
        input_channels : Number of input channels to the first residual block
        num_channels : Number of output channels for all residual blocks in this stage
        num_residuals : Number of residual blocks in this ResNet block
        first_block: Indicates whether this is the first ResNet block in the network. If True, the first residual block does not apply a 1x1 convolution or downsampling

    Returns:
        block:  A list of residual blocks forming a complete ResNet stage
    """
    layers = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # Uses a 1x1 convolution to match the number of channels and applies stride=2 for downsampling
            layers.append(Residual(input_channels, num_channels, use_1x1_conv=True, strides=2))
        else:
            # Standard residual block with identity shortcut (no downsampling)
            layers.append(Residual(num_channels, num_channels))

    return nn.Sequential(*layers)

class ResNet(ConvNet):
    def init_network(self):
        block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=64, 
                      kernel_size=7, 
                      stride=2, 
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1)
        )

        block2 = resnet_block(input_channels=64, 
                              num_channels=64, 
                              num_residuals=3, 
                              first_block=True)
        
        block3 = resnet_block(input_channels=64, 
                              num_channels=128, 
                              num_residuals=4)
        
        block4 = resnet_block(input_channels=128, 
                              num_channels=256, 
                              num_residuals=6)
        
        block5 = resnet_block(input_channels=256, 
                              num_channels=512, 
                              num_residuals=3)
        
        self.network = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            block5,

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=self.out_channels)
        )
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: ResNet-34"