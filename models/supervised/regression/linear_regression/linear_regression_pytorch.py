import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from ....base import Model

torch.manual_seed(42)

class LinearRegressionModule(nn.Module):
    """
    Linear regression model using PyTorch's nn.Module
    """
    def __init__(self) -> None:
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

        Returns:
            Tensor: Output of the linear layer after applying the weights and bias
        """
        return self.linear(x)
    
class LinearRegressionPytorch(Model):
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int) -> None:
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
            targets: torch.Tensor) -> None:
        """
        Trains the linear regression model on the input data

        Parameters:
            features: The input features for training
            targets: The target targets corresponding to the input features
        """
        targets = targets.unsqueeze(1).to(dtype=torch.float)
        self.model = LinearRegressionModule()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        for _ in range(self.number_of_epochs):
            predictions = self.model(features)  # Forward pass
            
            cost = func.mse_loss(predictions, targets)  # Compute mean squared error loss
            
            optimizer.zero_grad()  # Reset gradients
            cost.backward()  # Backpropagation
            optimizer.step()  # Update weights
        
        # Extract the learned parameters (weight and bias)
        params = list(self.model.parameters())
        self.weight = params[0].item()
        self.bias = params[1].item()
    
    def predict(self, 
                test_features: torch.Tensor, 
                test_targets: torch.Tensor = None):

        """
        Predict continuous target values for given samples

        Parameters:
            test_features: Test feature matrix of shape 
            test_targets: Test target values (optional, for evaluation)

        Returns:
            torch.Tensor: Predicted target values
        """
        
        predictions = (self.weight * test_features) + self.bias

        if test_targets is not None:
            mse, r2 = self.regression_evaluate(predictions, test_targets)
            print("MSE: {:.5f} R-squared: {:.5f}".format(mse, r2))

        return predictions
    
    def __str__(self) -> str:
        return "Linear Regression (Pytorch)"
