import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from base_model import ModelML

torch.manual_seed(42)

class LinearRegressionModule(nn.Module):
    """
    Linear regression model using PyTorch's nn.Module
    """
    def __init__(self):
        """
        Initializes the linear regression model by defining a single linear layer
        """
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer

        Parameters:
        x: Input features

        --------------------------------------------------
        Returns:
        Tensor: Output of the linear layer after applying the weights and bias
        """
        return self.linear(x)
    
class LinearRegressionPytorch(ModelML):
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int):
        """
        Initializes the Linear Regression model with the learning rate and number of epochs

        Parameters:
        learn_rate: The learning rate for the optimizer
        number_of_epochs: The number of training iterations to run
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Trains the linear regression model on the input data

        Parameters:
        features: The input features for training
        labels: The target labels corresponding to the input features
        """
        labels = labels.unsqueeze(1).to(dtype=torch.float)
        self.model = LinearRegressionModule()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        for _ in range(self.number_of_epochs):
            predictions = self.model(features)  # Forward pass
            
            cost = func.mse_loss(predictions, labels)  # Compute mean squared error loss
            
            optimizer.zero_grad()  # Reset gradients
            cost.backward()  # Backpropagation
            optimizer.step()  # Update weights
        
        # Extract the learned parameters (weight and bias)
        params = list(self.model.parameters())
        weight = params[0].item()
        bias = params[1].item()
    
        # Print the final state of the model
        print("Epoch: {}/{} Weight: {:.5f}, Bias: {:.5f} Cost: {:.5f}".format(
               self.number_of_epochs, self.number_of_epochs, weight, bias, cost))
    
    def __str__(self) -> str:
        return "Linear Regression (Pytorch)"
