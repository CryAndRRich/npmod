import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from ..base_model import ModelML

torch.manual_seed(42)

class SoftmaxRegressionModule(nn.Module):
    """
    Softmax Regression model using PyTorch's nn.Module for multi-class classification
    """
    def __init__(self, 
                 number_of_features: int, 
                 number_of_classes: int):
        """
        Initializes the softmax regression model by defining a linear layer

        Parameters:
        number_of_features: The number of input features for each data point
        number_of_classes: The number of classes for the output predictions
        """
        super().__init__()
        self.linear = nn.Linear(in_features=number_of_features, out_features=number_of_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the softmax regression model

        Parameters:
        x: The input features

        --------------------------------------------------
        Returns:
        torch.Tensor: Output of the softmax regression model (logits)
        """
        return self.linear(x)

class SoftmaxRegressionPytorch(ModelML):
    """
    Softmax Regression implemented using PyTorch with stochastic gradient descent optimization.
    """
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int, 
                 number_of_classes: int = 2) -> None:
        """
        Initializes the softmax regression model with the specified learning rate, 
        number of epochs, and number of output classes

        Parameters:
        learn_rate: The learning rate for the optimizer
        number_of_epochs: The number of training iterations
        number_of_classes: The number of output classes (default is 2 for binary classification)
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.number_of_classes = number_of_classes
    
    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Trains the softmax regression model on the input data

        Parameters:
        features: The input features for training
        labels: The true target labels corresponding to the input features
        """
        _, n = features.shape  # Get the number of features
        self.model = SoftmaxRegressionModule(n, self.number_of_classes)  # Initialize the Softmax regression model

        # Define the optimizer (Stochastic Gradient Descent)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        # Set the model to training mode
        self.model.train()

        # Perform training over the specified number of epochs
        for _ in range(self.number_of_epochs):
            predictions = self.model(features)  # Forward pass: Get predictions from the model

            # Compute the cross-entropy loss
            cost = func.cross_entropy(predictions, labels)

            optimizer.zero_grad()  # Reset gradients to avoid accumulation
            cost.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update model parameters using the optimizer

        # Set the model to evaluation mode after training is complete
        self.model.eval()
    
    def predict(self, 
                test_features: torch.Tensor, 
                test_labels: torch.Tensor) -> None:
        """
        Predicts the labels for the test data using the trained softmax regression model.

        Parameters:
        test_features: The input features for testing
        test_labels: The true target labels corresponding to the test features
        """
        with torch.no_grad():
            # Forward pass: Get predictions from the model
            predictions = self.model(test_features)

            # Get the class with the maximum probability for each data point (i.e., the predicted class)
            predictions = predictions.max(1)[1]

            # Convert predictions and test labels to numpy arrays
            predictions = predictions.detach().numpy()
            test_labels = test_labels.detach().numpy()

            # Evaluate the model using accuracy and F1-score
            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                   self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self) -> str:
        """
        Returns a string representation of the softmax regression model
        """
        return "Softmax Regression (Pytorch)"
