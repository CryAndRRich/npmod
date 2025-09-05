import torch
import torch.nn as nn
from ..cnn import ConvNet

class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution used in Xception

    This module applies a depthwise separable convolution consisting of a depthwise convolution
    followed by a pointwise convolution.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0) -> None:
        """
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride for the convolution
            padding: Padding for the convolution
        """
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, 
                                   out_channels=in_channels, 
                                   kernel_size=kernel_size, 
                                   stride=stride, 
                                   padding=padding, 
                                   groups=in_channels, 
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=1, 
                                   bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for depthwise separable convolution
        
        Parameters:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying separable convolution
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class XceptionBlock(nn.Module):
    """
    Xception Block with depthwise separable convolutions and residual connection

    This block repeats a series of separable convolutions and adds a shortcut (skip connection)
    to help with gradient flow
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 repeat: int, 
                 stride: int = 1, 
                 skip_connection: bool = True) -> None:
        """
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            repeat: Number of separable convolution layers to repeat
            stride: Stride for the first separable convolution in the block
            skip_connection: Whether to include a residual skip connection
        """
        super().__init__()
        self.skip_connection = skip_connection
        layers = []

        # First separable convolution may change spatial dimensions via stride
        layers.append(nn.ReLU(inplace=False))
        layers.append(SeparableConv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=3, 
                                      stride=stride, 
                                      padding=1))

        # Subsequent separable conv layers keep spatial dimensions constant.
        for _ in range(repeat - 1):
            layers.append(nn.ReLU(inplace=False))
            layers.append(SeparableConv2d(in_channels=out_channels, 
                                          out_channels=out_channels, 
                                          kernel_size=3, 
                                          stride=1, 
                                          padding=1))
        self.block = nn.Sequential(*layers)
        
        # Define the skip connection if needed
        if self.skip_connection:
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=1, 
                                      stride=stride, 
                                      bias=False)
                self.skip_bn = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.skip = None
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Xception Block
        
        Parameters:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor after processing through the Xception Block
        """
        out = self.block(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skip_bn(skip)
        else:
            skip = x
        return out + skip

class Xception(ConvNet):
    """
    Xception model uses depthwise separable convolutions and residual connections. 
    The network is divided into Entry Flow, Middle Flow, and Exit Flow
    """
    def init_network(self):
        # Entry Flow: Initial standard conv layers
        entry_flow = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False)
        )
        
        # Xception Blocks for Entry Flow (with downsampling)
        block1 = XceptionBlock(in_channels=64, 
                               out_channels=128, 
                               repeat=2, 
                               stride=1, 
                               skip_connection=True)
        block2 = XceptionBlock(in_channels=128, 
                               out_channels=256, 
                               repeat=2, 
                               stride=1, 
                               skip_connection=True)
        
        # Middle Flow: Series of Xception Blocks without downsampling
        middle_flow = nn.Sequential(
            XceptionBlock(in_channels=256, 
                          out_channels=256, 
                          repeat=2, 
                          stride=1, 
                          skip_connection=True),
            XceptionBlock(in_channels=256, 
                          out_channels=256, 
                          repeat=2, 
                          stride=1, 
                          skip_connection=True),
            XceptionBlock(in_channels=256, 
                          out_channels=256, 
                          repeat=2, 
                          stride=1, 
                          skip_connection=True)
        )
        
        # Exit Flow: Final Xception Block and separable convolution before classification
        exit_flow = nn.Sequential(
            XceptionBlock(in_channels=256, 
                          out_channels=512, 
                          repeat=2, 
                          stride=1, 
                          skip_connection=True),
            SeparableConv2d(in_channels=512, 
                            out_channels=512, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=False)
        )
        
        self.network = nn.Sequential(
            entry_flow,

            block1,
            block2,

            middle_flow,

            exit_flow,

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10)
        )
        
        self.network.apply(self.init_weights)

    def __str__(self) -> str:
        return "Convolutional Neural Networks: Xception"
