import torch.nn as nn
from ..cnn import ConvNet

class LeNet(ConvNet):
    def init_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=6, 
                      kernel_size=5),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, 
                      out_channels=16, 
                      kernel_size=5),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=120),
            nn.BatchNorm1d(num_features=120),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=120, out_features=84),
            nn.BatchNorm1d(num_features=84),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=84, out_features=self.out_channels)
        )
        self.network.apply(self.init_weights)
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: LeNet-5"