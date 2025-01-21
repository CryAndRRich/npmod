import torch
import torch.nn as nn
from base_model import ModelML

torch.manual_seed(42)

class PerceptionModule(nn.Module):
    """
    Perceptron model using PyTorch's nn.Module for binary classification
    """
    def __init__(self, n: int):
        """
        Initializes the perceptron model by defining a single linear layer

        Parameters:
        n: The number of input features
        """
        super().__init__()
        self.linear = nn.Linear(in_features=n, out_features=1)

    def heaviside_step(self, weighted_sum: torch.Tensor) -> torch.Tensor:
        """
        Applies the Heaviside step function to make binary predictions

        Parameters:
        weighted_sum: The weighted sum from the linear layer

        --------------------------------------------------
        Returns:
        torch.Tensor: Binary predictions (0 or 1) after applying the step function
        """
        # Convert weighted_sum into binary values (0 or 1)
        weighted_sum = [int(weight >= 0) for weight in weighted_sum]
        return torch.tensor(weighted_sum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the perceptron model

        Parameters:
        x: Input features

        --------------------------------------------------
        Returns:
        torch.Tensor: Binary predictions from the perceptron model
        """
        weighted_sum = self.linear(x)  # Compute the weighted sum
        return self.heaviside_step(weighted_sum)  # Apply the Heaviside step function

class PerceptronLearningPytorch(ModelML):
    """
    Perceptron learning algorithm implemented using PyTorch.
    """
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int) -> None:
        """
        Initializes the perceptron learning model with a given learning rate and number of epochs.

        Parameters:
        learn_rate: The learning rate for updating weights
        number_of_epochs: The number of training iterations
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Trains the perceptron model on the input features and labels.

        Parameters:
        features: The input features for training
        labels: The true target labels corresponding to the input features
        """
        _, n = features.shape  # Get the number of features
        self.model = PerceptionModule(n)  # Initialize the Perceptron module

        # Set the model to training mode
        self.model.train()

        # Perform training over the specified number of epochs
        for _ in range(self.number_of_epochs):
            cost = 0  # Initialize total cost for each epoch
            for x, y in zip(features, labels):
                # Forward pass: Get predictions from the model
                predictions = self.model(x)

                # Compute the error (difference between predictions and labels)
                error = predictions - y
                cost += error  # Accumulate the total cost

                # Get the current weights and bias
                weights = self.model.linear.weight
                bias = self.model.linear.bias

                # Update the weights and bias using the perceptron learning rule
                weights = weights - self.learn_rate * error * x
                bias = bias - self.learn_rate * error

                # Update the parameters in the model
                self.model.linear.weight = nn.Parameter(weights)
                self.model.linear.bias = nn.Parameter(bias)
        
        # Set the model to evaluation mode after training is complete
        self.model.eval()
    
    def predict(self, 
                test_features: torch.Tensor, 
                test_labels: torch.Tensor) -> None:
        """
        Predicts the labels for the test data using the trained perceptron model.

        Parameters:
        test_features: The input features for testing
        test_labels: The true target labels corresponding to the test features
        """
        with torch.no_grad():
            # Forward pass: Get predictions from the model
            predictions = self.model(test_features)
            predictions = predictions.detach().numpy()  # Convert predictions to numpy array
            test_labels = test_labels.detach().numpy()  # Convert labels to numpy array

            # Evaluate the model using accuracy and F1-score
            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                   self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self) -> str:
        """
        Returns a string representation of the perceptron model
        """
        return "Perceptron Learning Algorithm (Pytorch)"
