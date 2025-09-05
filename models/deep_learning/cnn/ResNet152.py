from typing import Tuple
import torch
import torch.nn as nn
from ..cnn import ConvNet

class Bottleneck(nn.Module):
    """
    Bottleneck Block for ResNet

    Each bottleneck block consists of three convolutional block:
      - 1x1 Conv to reduce the number of channels
      - 3x3 Conv for processing
      - 1x1 Conv to expand the number of channels (by the expansion factor)
    
    The default expansion factor is 4, meaning the output channels will be channels * 4
    """
    expansion: int = 4
    def __init__(self, 
                 input_channels: int, 
                 channels: int, 
                 stride: int = 1, 
                 downsample: nn.Module = None, 
                 **kwargs) -> None:
        """
        Parameters:
            input_channels: Number of input channels
            channels: Number of intermediate channels (output channels will be channels * expansion)
            stride: Stride for the 3x3 convolution
            downsample: If not None, used to match the dimensions of the identity connection
        """
        super().__init__(**kwargs)
        
        # Build the bottleneck network as a single sequential block
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=channels, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels, 
                      out_channels=channels, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels, 
                      out_channels=channels * self.expansion, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=channels * self.expansion)
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

def bottleneck_block(input_channels: int, 
                     channels: int, 
                     num_blocks: int, 
                     stride: int) -> Tuple[nn.Sequential, int]:
    """
    Creates a stage with multiple bottleneck blocks

    Parameters:
        input_channels: Number of input channels for the stage
        channels: Number of intermediate channels for the bottleneck block 
        num_blocks: Number of bottleneck blocks in the stage
        stride: Stride for the first block in the stage (for downsampling)
    
    Returns:
        A tuple consisting of:
         - An nn.Sequential containing the bottleneck blocks of the stage
         - The number of output channels after the stage (for the next stage)
    """
    blocks = []
    expansion = Bottleneck.expansion
    downsample = None

    # If stride is not 1 or input channels don't match output channels, apply downsampling
    if stride != 1 or input_channels != channels * expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=channels * expansion, 
                      kernel_size=1, 
                      stride=stride, 
                      bias=False),
            nn.BatchNorm2d(num_features=channels * expansion)
        )
    
    # First block may perform downsampling
    blocks.append(Bottleneck(input_channels, channels, stride, downsample))
    input_channels = channels * expansion

    # Subsequent blocks in the stage (stride always = 1, no downsampling)
    for _ in range(1, num_blocks):
        blocks.append(Bottleneck(input_channels, channels))
    
    return (nn.Sequential(*blocks), input_channels)

class ResNet152(ConvNet):
    def init_network(self):
        # Adjusted initial block for 28x28 input images:
        block1 = nn.Sequential(
            # Replace the original 7x7 conv with a 3x3 conv (stride=1) to better preserve spatial dimensions
            nn.Conv2d(in_channels=1, 
                      out_channels=16, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True)
            # Removed MaxPool2d to avoid excessive downsampling for small images
        )

        # Build the ResNet-152 stages using bottleneck blocks
        block2, input_channels = bottleneck_block(16, 16, num_blocks=2, stride=1)
        block3, input_channels = bottleneck_block(input_channels, 32, num_blocks=4, stride=2)
        block4, input_channels = bottleneck_block(input_channels, 64, num_blocks=6, stride=2)
        block5, input_channels = bottleneck_block(input_channels, 128, num_blocks=2, stride=2)

        self.network = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            block5,

            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=input_channels, out_features=10)
        )
        self.network.apply(self.init_weights)

    def __str__(self) -> str:
        return "Convolutional Neural Networks: ResNet-152"
