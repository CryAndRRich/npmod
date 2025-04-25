import torch.nn as nn
import torch.optim as optim
from ..cnn import Reshape, ConvNet
    
def vgg_block(num_convs: int, 
              in_channels: int, 
              out_channels: int,
              use_adaptive_pool: bool = False) -> nn.Sequential:
    """
    Implementation of one VGG block consists of a sequence of convolutional layers, 
    followed by a max pooling layer for spatial downsampling

    Parameters:
        num_convs: Number of convolutional layers
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_adaptive_pool: If True, the pooling layer is replaced with AdaptiveMaxPool2d
    
    Returns:
        block: One VGG block
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=3, 
                                padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    
    if use_adaptive_pool:
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
    else:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    block = nn.Sequential(*layers)
    return block

class VGG(ConvNet):
    def init_network(self):
        # A list of tuples (one per block), where each contains two values: 
        # the number of convolutional layers and the number of output channels
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        
        conv_layers = []
        in_channels = 1
        
        for i, (num_convs, out_channels) in enumerate(conv_arch):
            if i == len(conv_arch) - 1:
                block = vgg_block(num_convs, in_channels, out_channels, use_adaptive_pool=True)
            else:
                block = vgg_block(num_convs, in_channels, out_channels)
            conv_layers.append(block)
            in_channels = out_channels
        
        self.network = nn.Sequential(
            Reshape(),

            *conv_layers,

            nn.Flatten(),
            nn.Linear(in_features=512, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=10)
        )
        self.network.apply(self.init_weights)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: VGG-11"