import torch
import torch.nn as nn
import torch.optim as optim
from ..cnn import Reshape, ConvNet

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
                 strides: int = 1, 
                 **kwargs) -> None:
        """
        Parameters:
            input_channels: Number of input channels
            num_channels: Number of output channels
            use_1x1_conv: If True, applies a 1x1 convolution to the input in the skip connection to match the shape
            strides: Stride value for the first convolutional layer
        """
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=num_channels, 
                      kernel_size=3, 
                      padding=1, 
                      stride=strides),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=num_channels, 
                      out_channels=num_channels, 
                      kernel_size=3, 
                      padding=1),
            nn.BatchNorm2d(num_features=num_channels)
        )

        if use_1x1_conv:
            self.conv = nn.Conv2d(in_channels=input_channels, 
                                   out_channels=num_channels, 
                                   kernel_size=1, 
                                   stride=strides)
        else:
            # Skip connection (1x1 convolution for dimension matching)
            self.conv = None

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        
        # If 1x1 convolution is used, transform input x
        if self.conv:
            x = self.conv(x)

        y += x  # Add skip connection
        y = self.relu(y)
        return y

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
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # First residual block of a non-initial stage:
            # Uses a 1x1 convolution to match the number of channels and applies stride=2 for downsampling
            block.append(Residual(input_channels, num_channels, use_1x1_conv=True, strides=2))
        else:
            # Standard residual block with identity shortcut (no downsampling)
            block.append(Residual(num_channels, num_channels))

    return nn.Sequential(*block)

class ResNet34(ConvNet):
    def init_network(self):
        # Adjusted block1 for 28x28 input images:
        block1 = nn.Sequential(
            # Replace the original 7x7 conv with a 3x3 conv (stride=1) to better preserve spatial dimensions
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
            # Removed MaxPool2d to avoid excessive downsampling for small images
        )

        block2 = resnet_block(64, 64, 3, first_block=True)
        block3 = resnet_block(64, 128, 4)
        block4 = resnet_block(128, 256, 6)
        block5 = resnet_block(256, 512, 3)
        
        self.network = nn.Sequential(
            Reshape(),

            block1,
            block2,
            block3,
            block4,
            block5,

            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10)
        )
        self.network.apply(self.init_weights)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: ResNet-34"
