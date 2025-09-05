import torch.nn as nn
from ..cnn import ConvNet
    
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
        layers.append(nn.AdaptiveMaxPool2d(output_size=(1, 1)))
    else:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    block = nn.Sequential(*layers)
    return block

class VGG(ConvNet):
    def init_network(self):
        # A list of tuples (one per block), where each contains two values: 
        # the number of convolutional layers and the number of output channels
        conv_arch = ((1, 32), (1, 64), (2, 128), (2, 256), (2, 256))
        
        conv_layers = []
        in_channels = 1
        
        for i, (num_convs, out_channels) in enumerate(conv_arch):
            use_adaptive_pool = (i == len(conv_arch) - 1)
            block = vgg_block(num_convs=num_convs, 
                              in_channels=in_channels, 
                              out_channels=out_channels, 
                              use_adaptive_pool=use_adaptive_pool)
            conv_layers.append(block)
            in_channels = out_channels
        
        self.network = nn.Sequential(
            *conv_layers,

            nn.Flatten(),
            nn.Linear(in_features=in_channels, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=10)
        )
        self.network.apply(self.init_weights)
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: VGG-11"