import torch
import torch.nn as nn
from ..cnn import ConvNet

class RegNetBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 stride: int, 
                 group_width: int = 8) -> None:
        """
        Initialize the RegNet Block
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the block
            group_width: Width of each group in the grouped convolution
        """
        super().__init__()
        groups = max(1, out_channels // group_width)

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=1, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels, 
                               kernel_size=3,
                               stride=stride, 
                               padding=1, 
                               groups=groups, 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels, 
                               kernel_size=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=1, 
                          stride=stride, 
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + self.downsample(identity)
        return self.relu(out)


class RegNet(ConvNet):
    def init_network(self):
        stem_channels = 32
        stage_channels = [24, 56, 152, 368]
        stage_blocks = [1, 1, 4, 7]

        layers = [
            nn.Conv2d(in_channels=3, 
                      out_channels=stem_channels,
                      kernel_size=3, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1)
        ]

        in_channels = stem_channels
        for out_channels, blocks in zip(stage_channels, stage_blocks):
            for i in range(blocks):
                stride = 2 if i == 0 else 1
                layers.append(RegNetBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          stride=stride))
                in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=in_channels, out_features=self.out_channels))

        self.network = nn.Sequential(*layers)

    def __str__(self):
        return "Convolutional Neural Networks: RegNetX-200MF"