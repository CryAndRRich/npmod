import torch
import torch.nn as nn
from ..cnn import ConvNet


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel Shuffle operation
    
    Parameters:
        x: Input tensor
        groups: Number of groups to divide the channels into
    
    Returns:
        x: Shuffled tensor
    """
    N, C, H, W = x.size()
    assert C % groups == 0, "Channels must be divisible by groups"
    channels_per_group = C // groups

    # reshape
    x = x.view(N, groups, channels_per_group, H, W)
    x = x.transpose(1, 2).contiguous()
    # flatten
    x = x.view(N, C, H, W)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int, 
                 groups: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.groups = groups

        out_channels_minus = out_channels if stride == 1 else out_channels - in_channels
        mid_channels = out_channels_minus // 4

        # 1x1 Grouped Conv
        self.gconv1 = nn.Conv2d(in_channels=in_channels, 
                                out_channels=mid_channels, 
                                kernel_size=1, 
                                stride=1, 
                                padding=0, 
                                groups=groups, 
                                bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Depthwise Conv
        self.dwconv = nn.Conv2d(in_channels=mid_channels, 
                                out_channels=mid_channels, 
                                kernel_size=3, 
                                stride=stride, 
                                padding=1,
                                groups=mid_channels, 
                                bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=mid_channels)

        # 1x1 Grouped Conv
        self.gconv2 = nn.Conv2d(in_channels=mid_channels, 
                                out_channels=out_channels_minus, 
                                kernel_size=1, 
                                stride=1, 
                                padding=0, 
                                groups=groups, 
                                bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels_minus)

        # Shortcut
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.relu(self.bn1(self.gconv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.dwconv(out))
        out = self.bn3(self.gconv2(out))

        if self.stride == 1:
            out = out + residual
        else:
            out = torch.cat([out, self.shortcut(residual)], 1)

        return self.relu(out)


class ShuffleNet(ConvNet):
    def init_network(self):
        groups = 2  
        out_channels = [24, 48, 96, 192]

        layers = []
        # First conv
        layers.append(nn.Conv2d(in_channels=1, 
                                out_channels=out_channels[0], 
                                kernel_size=3, 
                                stride=1, 
                                padding=1, 
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=out_channels[0]))
        layers.append(nn.ReLU(inplace=True))

        in_channel = out_channels[0]
        for out_channel in out_channels[1:]:
            layers.append(ShuffleBlock(in_channels=in_channel, 
                                       out_channels=out_channel, 
                                       stride=2, 
                                       groups=groups))
            in_channel = out_channel

        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=in_channel, out_features=10))

        self.network = nn.Sequential(*layers)
        self.network.apply(self.init_weights)

    def __str__(self):
        return "Convolutional Neural Networks: ShuffleNetV1"
