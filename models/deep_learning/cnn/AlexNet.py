import torch.nn as nn
from ..cnn import ConvNet
    
class AlexNet(ConvNet):
    def init_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=32, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, 
                      out_channels=128, 
                      kernel_size=3, 
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=256 * 3 * 3, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=512, out_features=10)
        )
        self.network.apply(self.init_weights)
    
    def __str__(self) -> str:
        return "Convolutional Neural Networks: AlexNet"
