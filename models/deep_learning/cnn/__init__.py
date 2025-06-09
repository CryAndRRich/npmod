import torch
import torch.nn as nn

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

# Since CNNs primarily operate with 224x224 images (ImageNet dataset), 
# the model architectures in this CNN folder will be modified 
# to suit 28x28 images for convenient training
class ConvNet():
    def __init__(self,
                 learn_rate: float, 
                 number_of_epochs: int) -> None:
        """
        Initializes the Convolutional Neural Networks

        Parameters:
            learn_rate: The learning rate for the network update
            number_of_epochs: The number of training iterations
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def init_network(self) -> None:
        """Initialize the network, optimizer and loss function"""
        pass

    def init_weights(self, m) -> None:
        """Initialize the model parameters using the Xavier initializer"""
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def fit(self, 
            features: torch.Tensor, 
            targets: torch.Tensor) -> None:
        """
        Train the network using the provided dataset

        Parameters:
            features: Input matrix
            targets: True targets containing class indices
        """
        self.init_network()

        self.network.train()
        for _ in range(self.number_of_epochs):
            self.optimizer.zero_grad()
            outputs = self.network(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
    def predict(self, test_features: torch.Tensor) -> torch.Tensor:
        """
        Makes predictions on the test set and evaluates the network

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        self.network.eval()
        with torch.no_grad():
            logits = self.network(test_features)
            predictions = torch.argmax(logits, dim=1)

        return predictions 

from .alexnet import AlexNet
from .densenet import DenseNet
from .googlenet import GoogLeNet
from .lenet import LeNet
from .nasnet import NASNet
from .nin import NiN
from .resnet34 import ResNet34
from .resnet152 import ResNet152
from .resnext import ResNeXt
from .vgg import VGG
from .wideresnet import WideResNet
from .xception import Xception