import torch.nn as nn
import torch.optim as optim
from ..CNN import ConvNet

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    
class AlexNet(ConvNet):
    def init_network(self):
        self.network = nn.Sequential(
            Reshape(),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True), 
            
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Flatten(),
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=4096, out_features=10)
        )
        self.network.apply(self.init_weights)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def __str__(self):
        return "Convolutional Neural Networks: AlexNet"