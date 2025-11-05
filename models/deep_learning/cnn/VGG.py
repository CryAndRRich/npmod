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
        conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
        conv_layers = []
        in_channels = 3
        
        for num_convs, out_channels in conv_arch:
            block = vgg_block(num_convs=num_convs, 
                              in_channels=in_channels, 
                              out_channels=out_channels)
            conv_layers.append(block)
            in_channels = out_channels
        
        self.network = nn.Sequential(
            *conv_layers,

            nn.Flatten(),
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=self.out_channels)
        )
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: VGG-16"