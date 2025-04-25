import torch
import torch.nn as nn
import torch.optim as optim
from ..cnn import Reshape, ConvNet

def conv_block(in_channels: int, 
               num_channels: int) -> nn.Sequential:
    """
    Constructs a convolutional block

    Parameters:
        in_channels: Number of input channels
        num_channels: Number of output channels
    
    Returns:
        block: The convolutional block
    """
    layers = []
    layers.append(nn.BatchNorm2d(num_features=in_channels))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(in_channels=in_channels, 
                            out_channels=num_channels, 
                            kernel_size=3, 
                            padding=1))
    
    block = nn.Sequential(*layers)
    return block

class DenseBlock(nn.Module):
    """
    Dense Block for DenseNet

    This block contains multiple convolutional layers (conv_block) where the output of each layer 
    is concatenated with its input. This dense connectivity helps improve feature propagation
    """
    def __init__(self, 
                 num_convs: int, 
                 input_channels: int, 
                 num_channels: int, 
                 **kwargs) -> None:
        """
        Parameters:
            num_convs: Number of convolution layers in the dense block
            input_channels: Number of input channels to the block
            num_channels: Growth rate (number of output channels for each convolution)
        """
        super().__init__(**kwargs)
        layers = []
        # Create each convolution layer; the input channels grow as outputs are concatenated
        for i in range(num_convs):
            layers.append(conv_block(input_channels + num_channels * i, num_channels))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Dense Block
        
        Parameters:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor after dense connections
        """
        # For each block, compute output and concatenate it with input along the channel dimension.
        for block in self.net:
            y = block(x)
            # Concatenate the input and output of each block on the channel dimension
            x = torch.cat((x, y), dim=1)
        return x

def transition_block(input_channels: int, 
                     num_channels: int) -> nn.Sequential:
    """
    Transition Block for DenseNet

    This block applies Batch Normalization, ReLU, a 1x1 convolution to reduce the number of channels,
    and then average pooling to reduce the spatial dimensions

    Parameters:
        input_channels: Number of input channels
        num_channels: Number of output channels after the 1x1 convolution
    
    Returns:
        block: The transition block
    """
    layers = []
    layers.append(nn.BatchNorm2d(num_features=input_channels))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(in_channels=input_channels, 
                            out_channels=num_channels, 
                            kernel_size=1))
    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
    
    block = nn.Sequential(*layers)
    return block

class DenseNet(ConvNet):
    """
    DenseNet consists of an initial convolution layer, followed by a series of
    Dense Blocks and Transition Blocks. The final layers perform classification
    """
    def init_network(self):
        # Set initial number of channels and growth rate for the dense blocks
        num_channels, growth_rate = 64, 32
        # Define number of convolution layers in each dense block
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        layers = []
        # Build the dense blocks with interleaved transition blocks
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            layers.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            # Add a transition block after each dense block except the last one
            if i != len(num_convs_in_dense_blocks) - 1:
                layers.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        blocks = nn.Sequential(*layers)

        # Modified initial layers for 28x28 images:
        # - Use a smaller kernel and stride to preserve spatial dimensions.
        self.network = nn.Sequential(
            Reshape(), 
            
            nn.Conv2d(in_channels=1, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            blocks,
            
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(in_features=num_channels, out_features=10)
        )

        self.network.apply(self.init_weights)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: DenseNet-121"
