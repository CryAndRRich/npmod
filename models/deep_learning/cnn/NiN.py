import torch.nn as nn
from ..cnn import ConvNet

def nin_block(in_channels: int,
              out_channels: int,
              kernel_size: int,
              stride: int,
              padding: int) -> nn.Sequential:
    """
    Creates a NiN block consisting of:
      - A main convolution with the given kernel size, stride, and padding,
      - Followed by two successive 1x1 convolutions for nonlinear feature extraction
    
    Each convolution is followed by a ReLU activation
    
    Parameters:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for the main convolution
        stride: Stride for the main convolution
        padding: Padding for the main convolution
    
    Returns:
        nn.Sequential: The constructed NiN block.
    """
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(in_channels=out_channels,
                  out_channels=out_channels,
                  kernel_size=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(in_channels=out_channels,
                  out_channels=out_channels,
                  kernel_size=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )
    
    return block

class NiN(ConvNet):
    def init_network(self):
        block1 = nin_block(in_channels=3, 
                           out_channels=96, 
                           kernel_size=11, 
                           stride=4, 
                           padding=2)
        
        block2 = nin_block(in_channels=96, 
                           out_channels=256, 
                           kernel_size=5, 
                           stride=2, 
                           padding=2)
        
        block3 = nin_block(in_channels=256, 
                           out_channels=384, 
                           kernel_size=3, 
                           stride=1, 
                           padding=1)
        
        block4 = nin_block(in_channels=384, 
                           out_channels=self.out_channels, 
                           kernel_size=3, 
                           stride=1, 
                           padding=1)
        
        self.network = nn.Sequential(
            block1,
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1),
            
            block2,
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1),
            
            block3,
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1),
            nn.Dropout2d(p=0.5),
            
            block4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        
    def __str__(self) -> str:
        return "Convolutional Neural Networks: NiN"