from typing import Tuple
import torch
import torch.nn as nn
from ..cnn import ConvNet

class ResNeXtBottleneck(nn.Module):
    """
    Bottleneck Block for ResNeXt

    This block consists of three convolutional layers aggregated into a sequential network:
      - A 1x1 convolution to reduce the number of channels
      - A 3x3 grouped convolution (with cardinality groups) for processing
      - A 1x1 convolution to expand the number of channels by the expansion factor
    
    The default expansion factor is 4
    """
    expansion: int = 4
    def __init__(self, 
                 in_channels: int, 
                 channels: int, 
                 stride: int = 1, 
                 cardinality: int = 32, 
                 base_width: int = 4, 
                 downsample: nn.Module = None, 
                 **kwargs) -> None:
        """
        Parameters:
            in_channels: Number of input channels
            channels: Number of intermediate channels before expansion.
                      The final output channels will be channels * expansion
            stride: Stride for the 3x3 grouped convolution
            cardinality: Number of groups for the grouped convolution
            base_width: Base width for each group
            downsample: If not None, applies downsampling on the shortcut branch
        """
        super().__init__(**kwargs)
        
        # Calculate the inner width for the grouped convolution
        inner_channels = cardinality * base_width
        
        # Build the bottleneck network as a single sequential block
        self.network = nn.Sequential(
            # 1x1 convolution to reduce dimensions
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=inner_channels, 
                      kernel_size=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=inner_channels),
            nn.ReLU(inplace=True),
            
            # 3x3 grouped convolution
            nn.Conv2d(in_channels=inner_channels, 
                      out_channels=inner_channels, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1, 
                      groups=cardinality, 
                      bias=False),
            nn.BatchNorm2d(num_features=inner_channels),
            nn.ReLU(inplace=True),
            
            # 1x1 convolution to expand dimensions
            nn.Conv2d(in_channels=inner_channels, 
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
        
        y += x
        y = self.relu(y)
        return y

def resnext_block(input_channels: int, 
                  channels: int, 
                  num_blocks: int, 
                  stride: int, 
                  cardinality: int, 
                  base_width: int) -> Tuple[nn.Sequential, int]:
    """
    Creates a block with multiple ResNeXt bottleneck blocks

    Parameters:
        input_channels: Number of input channels for the block
        channels: Number of intermediate channels for the blocks in the block
        num_blocks: Number of ResNeXt bottleneck blocks in the block
        stride: Stride for the first block (for downsampling)
        cardinality: Number of groups for the grouped convolution
        base_width: Base width for each group
    
    Returns:
        A tuple consisting of:
         - An nn.Sequential containing the blocks
         - The number of output channels after the block (for the next block)
    """
    downsample = None
    expansion = ResNeXtBottleneck.expansion
    if stride != 1 or input_channels != channels * expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=channels * expansion, 
                      kernel_size=1, 
                      stride=stride, 
                      bias=False),
            nn.BatchNorm2d(num_features=channels * expansion)
        )
    
    layers = []
    layers.append(ResNeXtBottleneck(in_channels=input_channels, 
                                    channels=channels, 
                                    stride=stride, 
                                    cardinality=cardinality, 
                                    base_width=base_width, 
                                    downsample=downsample))
    input_channels = channels * expansion
    for _ in range(1, num_blocks):
        layers.append(ResNeXtBottleneck(in_channels=input_channels, 
                                        channels=channels, 
                                        stride=1, 
                                        cardinality=cardinality, 
                                        base_width=base_width))
    
    block = nn.Sequential(*layers)
    return (block, input_channels)

class ResNeXt(ConvNet):
    def init_network(self):
        """
        Initializes the ResNeXt-50 (32x4d) network adapted for 28x28 images

        The architecture follows:
          - An initial 3x3 convolution block (without max pooling) to better suit small input images
          - Four blocks of ResNeXt bottleneck blocks
          - An adaptive average pooling followed by a fully connected layer
        """
        block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        cardinality = 8
        base_width = 4
        
        block2, in_channels = resnext_block(input_channels=64, 
                                            channels=32, 
                                            num_blocks=3, 
                                            stride=1, 
                                            cardinality=cardinality, 
                                            base_width=base_width)
        block3, in_channels = resnext_block(input_channels=in_channels, 
                                            channels=64, 
                                            num_blocks=4, 
                                            stride=2, 
                                            cardinality=cardinality, 
                                            base_width=base_width)
        block4, in_channels = resnext_block(input_channels=in_channels, 
                                            channels=128, 
                                            num_blocks=6, 
                                            stride=2, 
                                            cardinality=cardinality, 
                                            base_width=base_width)
        block5, in_channels = resnext_block(input_channels=in_channels, 
                                            channels=256, 
                                            num_blocks=3, 
                                            stride=1, 
                                            cardinality=cardinality, 
                                            base_width=base_width)
        
        self.network = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            block5,

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=in_channels, out_features=10)
        )
        
        self.network.apply(self.init_weights)

    def __str__(self) -> str:
        return "Convolutional Neural Networks: ResNeXt-50"
