from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ..base import Model

class MLPModule(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 n_layers: int, 
                 n_neurons: List[int], 
                 output_dim: int) -> None:
        """
        Parameters:
            input_dim: Number of input features (including bias if needed)
            n_layers: Number of hidden layers
            n_neurons: Number of neurons in each hidden layer
            output_dim: Number of output classes
        """
        super().__init__()
        
        layers = []
        # Input layer -> First hidden layer
        layers.append(nn.Linear(input_dim, n_neurons[0]))
        layers.append(nn.Sigmoid())
        
        # Additional hidden layers
        assert len(n_neurons) == n_layers, \
            "Number of neurons in each hidden layer must equal to Number of hidden layers"
        for i in range(n_layers - 1):
            layers.append(nn.Linear(n_neurons[i], n_neurons[i + 1]))
            layers.append(nn.Sigmoid())
        
        # Output layer
        layers.append(nn.Linear(n_neurons[-1], output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network

        --------------------------------------------------
        Parameters:
            x: Input tensor

        --------------------------------------------------
        Returns:
            torch.Tensor: Logits (output before softmax)
        """
        return self.network(x)

class MLPPytorch(Model):
    def __init__(self,
                 batch_size: int,
                 learn_rate: float, 
                 number_of_epochs: int,
                 n_layers: int,
                 n_neurons: List[int]) -> None:
        """
        Initializes the Multilayer Perceptron model

        --------------------------------------------------
        Parameters:
            batch_size: Size of a training mini-batch
            learn_rate: The learning rate for the model update
            number_of_epochs: The number of training iterations
            n_layers: Number of hidden layers
            n_neurons: Number of neurons in each hidden layer
        """
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.n_layers = n_layers
        self.n_neurons = n_neurons

    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Train the model using the provided dataset

        Parameters:
            features: Input matrix
            labels: True labels containing class indices
        """
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = MLPModule(input_dim=features.shape[1], 
                          n_layers=self.n_layers,
                          n_neurons=self.n_neurons,
                          output_dim=10)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for _ in range(self.number_of_epochs):
            for batch_features, batch_labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
    
    def predict(self, 
                test_features: torch.Tensor, 
                test_labels: torch.Tensor,
                get_accuracy: bool = True) -> torch.Tensor:
        """
        Makes predictions on the test set and evaluates the model

        --------------------------------------------------
        Parameters:
            test_features: The input features for testing
            test_labels: The true target labels corresponding to the test features
            get_accuracy: If True, calculates and prints the accuracy of predictions

        --------------------------------------------------
        Returns:
            predictions: The prediction labels
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(test_features)
            predictions = torch.argmax(logits, dim=1)

            if get_accuracy:
                # Evaluate accuracy and F1 score
                accuracy, f1 = self.evaluate(predictions, test_labels)
                print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                    self.number_of_epochs, self.number_of_epochs, accuracy, f1))
        
        return predictions 
    
    def __str__(self):
        return "Multilayer Perceptron (Pytorch)"