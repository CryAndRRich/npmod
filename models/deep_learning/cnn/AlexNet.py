import torch.nn as nn
from ..cnn import ConvNet
    
class AlexNet(ConvNet):
    def init_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=64, 
                      kernel_size=11, 
                      stride=4, 
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=64, 
                      out_channels=192, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=192, 
                      out_channels=384, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, 
                      out_channels=256, 
                      kernel_size=3, 
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, 
                      out_channels=256, 
                      kernel_size=3, 
                      padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Flatten(),
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=4096, out_features=self.out_channels)
        )
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: AlexNet"