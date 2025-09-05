import torch
import torch.nn as nn
from ..cnn import ConvNet

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution used in MobileNetV1
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1) -> None:
        """
        Initialize the Depthwise Separable Convolution
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the depthwise convolution
        """
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, 
                                   out_channels=in_channels, 
                                   kernel_size=3, 
                                   stride=stride,
                                   padding=1, 
                                   groups=in_channels, 
                                   bias=False)
        self.dw_bn = nn.BatchNorm2d(num_features=in_channels)
        self.dw_relu = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=1, 
                                   bias=False)
        self.pw_bn = nn.BatchNorm2d(num_features=out_channels)
        self.pw_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Depthwise Separable Convolution"""
        x = self.dw_relu(self.dw_bn(self.depthwise(x)))
        x = self.pw_relu(self.pw_bn(self.pointwise(x)))
        return x


class MobileNet(ConvNet):
    def init_network(self):
        layers = []

        width_mult = 0.25  # shrink channels to 1/4
        def c(channels: int) -> int: 
            return max(8, int(channels * width_mult))

        # First conv
        layers.append(nn.Conv2d(in_channels=1, 
                                out_channels=c(32), 
                                kernel_size=3, 
                                stride=1, 
                                padding=1, 
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=c(32)))
        layers.append(nn.ReLU(inplace=True))

        # Depthwise separable conv blocks
        cfg = [
            (64, 1), (128, 2), (128, 1),
            (256, 2), (256, 1),
            (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
            (1024, 2), (1024, 1)
        ]

        in_channels = c(32)
        for out_channels, stride in cfg:
            layers.append(DepthwiseSeparableConv(in_channels=in_channels, 
                                                 out_channels=c(out_channels), 
                                                 stride=stride))
            in_channels = c(out_channels)

        # Classifier
        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=c(1024), out_features=10))

        self.network = nn.Sequential(*layers)
        self.network.apply(self.init_weights)

    def __str__(self):
        return "Convolutional Neural Networks: MobileNetV1"
