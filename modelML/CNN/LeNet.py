import torch.nn as nn
import torch.optim as optim
from ..CNN import ConvNet

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

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
    
    def __str__(self):
        return "Convolutional Neural Networks: LeNet5"