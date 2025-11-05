from typing import Tuple
import torch
import torch.nn as nn
from ..cnn import ConvNet

class WRNBlock(nn.Module):
    """
    Basic residual block for Wide ResNet

    This block consists of two 3x3 convolutional block:
      - The first conv reduces the spatial dimension (if stride > 1) and is followed by BatchNorm, ReLU, and dropout
      - The second conv is followed by BatchNorm
    
    A shortcut (identity connection) is added to the output. 
    If the input and output dimensions differ, a 1x1 convolution is used for matching
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1, 
                 dropout_rate: float = 0.3, 
                 **kwargs) -> None:
        """
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolutional layer
            dropout_rate: Dropout rate applied after the first convolution
        """
        super().__init__(**kwargs)
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Conv2d(in_channels=out_channels, 
                      out_channels=out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Define the shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=1, 
                          stride=stride, 
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)
        shortcut = self.shortcut(x)
        y += shortcut
        y = self.relu(y)
        return y

def wrn_block(input_channels: int, 
              out_channels: int, 
              num_blocks: int, 
              stride: int, 
              dropout_rate: float = 0.3) -> Tuple[nn.Sequential, int]:
    """
    Creates a stage with multiple WRNBlock blocks

    Parameters:
        input_channels: Number of input channels for the stage
        out_channels: Number of output channels for the blocks in the stage
        num_blocks: Number of WRNBlock blocks in the stage
        stride: Stride for the first block in the stage (for downsampling)
        dropout_rate: Dropout rate used in the blocks
    
    Returns:
        A tuple consisting of:
         - An nn.Sequential containing the blocks of the stage
         - The number of output channels after the stage (for the next stage)
    """
    block = []

    block.append(WRNBlock(in_channels=input_channels, 
                          out_channels=out_channels, 
                          stride=stride, 
                          dropout_rate=dropout_rate))
    input_channels = out_channels

    for _ in range(1, num_blocks):
        block.append(WRNBlock(in_channels=input_channels, 
                              out_channels=out_channels, 
                              stride=1, 
                              dropout_rate=dropout_rate))
    
    return nn.Sequential(*block), input_channels

class WideResNet(ConvNet):
    def init_network(self):
        depth = 28
        widen_factor = 10

        n = (depth - 4) // 6 

        in_channels = 64

        block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=in_channels,
                      kernel_size=7, 
                      stride=2, 
                      padding=3, 
                      bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1)
        )

        block2, out_channels = wrn_block(input_channels=in_channels, 
                                         out_channels=64 * widen_factor, 
                                         num_blocks=n, 
                                         stride=1)
        
        block3, out_channels = wrn_block(input_channels=out_channels, 
                                         out_channels=128 * widen_factor, 
                                         num_blocks=n, 
                                         stride=2)
        
        block4, out_channels = wrn_block(input_channels=out_channels, 
                                         out_channels=256 * widen_factor, 
                                         num_blocks=n, 
                                         stride=2)
        
        block5, out_channels = wrn_block(input_channels=out_channels, 
                                         out_channels=512 * widen_factor, 
                                         num_blocks=n, 
                                         stride=2)
        
        self.network = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            block5,

            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=out_channels, out_features=self.out_channels)
        )

    def __str__(self) -> str:
        return "Convolutional Neural Networks: WideResNet-28-10"