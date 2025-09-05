import torch
import torch.nn as nn
from ..cnn import ConvNet

class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution used in NASNet

    This module applies a depthwise separable convolution consisting of a depthwise convolution
    followed by a pointwise convolution
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
        self.relu = nn.ReLU(inplace=True)

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

class NormalCell(nn.Module):
    """
    Normal Cell in NASNet

    This cell preserves the spatial dimensions of the input while extracting features.
    It is constructed with two branches that are summed and then passed through a ReLU
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int) -> None:
        """
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.branch1 = nn.Sequential(
            SeparableConv2d(in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1),
            SeparableConv2d(in_channels=out_channels, 
                            out_channels=out_channels, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1)
        )
        self.branch2 = SeparableConv2d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size=3, 
                                       stride=1, 
                                       padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Normal Cell
        
        Parameters:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor after processing through the Normal Cell
        """
        out = self.branch1(x) + self.branch2(x)
        return self.relu(out)

class ReductionCell(nn.Module):
    """
    Reduction Cell in NASNet

    This cell reduces the spatial dimensions of the input while increasing the number of channels.
    It uses stride-2 operations in its separable convolutions and pooling to downsample the feature maps
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int) -> None:
        """
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.branch1 = SeparableConv2d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size=3, 
                                       stride=2, 
                                       padding=1)
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=1, 
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = SeparableConv2d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size=3, 
                                       stride=2, 
                                       padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Reduction Cell
        
        Parameters:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor with reduced spatial dimensions
        """
        out = self.branch1(x) + self.branch2(x) + self.branch3(x)
        return self.relu(out)

class NASNet(ConvNet):
    """
    Full NASNet Architecture

    The architecture consists of:
      - A stem that acts as an initial reduction cell
      - A series of Normal Cells
      - Interleaved Reduction Cells
      - Global Average Pooling and a fully-connected layer for classification
    """
    def init_network(self):
        num_cells = 2

        # Stem: Acts as an initial reduction cell to downsample the input
        stem = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=16, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True)
        )

        cells1 = nn.Sequential(*[NormalCell(in_channels=16 if i == 0 else 32, out_channels=32) 
                                 for i in range(num_cells)])
        reduction1 = ReductionCell(in_channels=32, out_channels=64)
        cells2 = nn.Sequential(*[NormalCell(in_channels=64, out_channels=64) 
                                 for _ in range(num_cells)])
        reduction2 = ReductionCell(in_channels=64, out_channels=128)
        cells3 = nn.Sequential(*[NormalCell(in_channels=128, out_channels=128) 
                                 for _ in range(num_cells)])
        
        self.network = nn.Sequential(
            stem,

            cells1,
            reduction1,

            cells2,
            reduction2,

            cells3,

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=10)
        )
        
        self.network.apply(self.init_weights)

    def __str__(self) -> str:
        return "Convolutional Neural Networks: NASNet"
