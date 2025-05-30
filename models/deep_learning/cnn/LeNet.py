import torch.nn as nn
import torch.optim as optim
from ..cnn import Reshape, ConvNet

class LeNet(ConvNet):
    def init_network(self):
        self.network = nn.Sequential(
            Reshape(),

            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(num_features=6),
            nn.Tanh(),

            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(num_features=16),
            nn.Tanh(),

            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=120),
            nn.BatchNorm1d(num_features=120),
            nn.Tanh(),

            nn.Linear(in_features=120, out_features=84),
            nn.BatchNorm1d(num_features=84),
            nn.Tanh(),

            nn.Linear(in_features=84, out_features=10)
        )
        self.network.apply(self.init_weights)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: LeNet-5"