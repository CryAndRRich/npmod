import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ..base_model import ModelML

class ConvNet(ModelML):
    def __init__(self,
                 batch_size: int,
                 learn_rate: float, 
                 number_of_epochs: int) -> None:
        """
        Initializes the Convolutional Neural Networks

        --------------------------------------------------
        Parameters:
            batch_size: Size of a training mini-batch
            learn_rate: The learning rate for the network update
            number_of_epochs: The number of training iterations
        """
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def init_network(self):
        """Initialize the network, optimizer and loss function"""
        self.network = nn.Sequential()
        self.network.apply(self.init_weights)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, m):
        """Initialize the model parameters using the Xavier initializer"""
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    
    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Train the network using the provided dataset

        Parameters:
            features: Input matrix
            labels: True labels containing class indices
        """
        self.init_network()

        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.network.train()
        for _ in range(self.number_of_epochs):
            for batch_features, batch_labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.network(batch_features)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
    
    def predict(self, 
                test_features: torch.Tensor, 
                test_labels: torch.Tensor,
                get_accuracy: bool = True) -> torch.Tensor:
        """
        Makes predictions on the test set and evaluates the network

        --------------------------------------------------
        Parameters:
            test_features: The input features for testing
            test_labels: The true target labels corresponding to the test features
            get_accuracy: If True, calculates and prints the accuracy of predictions

        --------------------------------------------------
        Returns:
            predictions: The prediction labels
        """
        self.network.eval()
        with torch.no_grad():
            logits = self.network(test_features)
            predictions = torch.argmax(logits, dim=1)

            if get_accuracy:
                # Evaluate accuracy and F1 score
                accuracy, f1 = self.evaluate(predictions, test_labels)
                print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                    self.number_of_epochs, self.number_of_epochs, accuracy, f1))
        
        return predictions 
