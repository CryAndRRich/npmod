import torch
import torch.optim as optim
import torch.nn as nn

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
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        """
        Trains the network on the training set
        
        Parameters:
            train_loader: The DataLoader for training data
            verbose: If True, prints training progress
        """
        self.init_network()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()
    
        self.network.train()
        for epoch in range(self.number_of_epochs):
            running_loss = 0.0
            for features, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.network(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
    
                running_loss += loss.item()
    
            if verbose:
                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{self.number_of_epochs}], Loss: {avg_loss:.4f}")
    
    def predict(self, test_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Makes predictions on the test set and evaluates the network

        Parameters:
            test_loader: The DataLoader for testing

        Returns:
            predictions: The prediction targets
            labels: The true targets
        """
        self.network.eval()
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for features, targets in test_loader:
                logits = self.network(features)
                preds = torch.argmax(logits, dim=1)
    
                all_preds.append(preds)
                all_labels.append(targets)
    
        predictions = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
    
        return predictions, labels

from .LeNet import LeNet             
from .AlexNet import AlexNet       
from .NiN import NiN                
from .VGG import VGG             
from .GoogLeNet import GoogLeNet    
from .ResNet34 import ResNet34  
from .ResNet152 import ResNet152   
from .SqueezeNet import SqueezeNet   
from .ResNeXt import ResNeXt         
from .Xception import Xception   
from .DenseNet import DenseNet     
from .WideResNet import WideResNet   
from .MobileNet import MobileNet     
from .NASNet import NASNet         
from .ShuffleNet import ShuffleNet    
from .EfficientNet import EfficientNet
from .RegNet import RegNet           
from .GhostNet import GhostNet       
from .MicroNet import MicroNet       
