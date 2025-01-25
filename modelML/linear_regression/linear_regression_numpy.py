from typing import Tuple
import numpy as np
from ..base_model import ModelML

def cost_function(features: np.ndarray, 
                  labels: np.ndarray, 
                  weight: float, 
                  bias: float) -> float:
    """
    Computes the mean squared error cost for linear regression

    Parameters:
    features: The input feature values 
    labels: The target labels corresponding to the input features 
    weight: The current weight value of the model
    bias: The current bias value of the model

    --------------------------------------------------
    Returns:
    avg_cost: The average cost (mean squared error) for the current weight and bias
    """
    m = features.shape[0]
    total_cost = 0

    # Compute the total squared error
    for i in range(m):
        x = features[i]
        y = labels[i]
        total_cost += (y - (weight * x + bias)) ** 2
    
    # Average cost over all data points
    avg_cost = total_cost / m
    return avg_cost

def gradient_descent(features: np.ndarray, 
                     labels: np.ndarray, 
                     weight: float, 
                     bias: float, 
                     learn_rate: float) -> Tuple[float, float]:
    """
    Performs one step of gradient descent to update the model's weight and bias

    Parameters:
    features: The input feature values 
    labels: The target labels corresponding to the input features 
    weight: The current weight value of the model
    bias: The current bias value of the model
    learn_rate: The learning rate for gradient descent

    --------------------------------------------------
    Returns:
    weight: The updated weight value after one step of gradient descent
    bias: The updated bias value after one step of gradient descent
    """
    m = features.shape[0]

    weight_gradient = 0
    bias_gradient = 0

    # Calculate gradients for weight and bias
    for i in range(m):
        x = features[i]
        y = labels[i]

        weight_gradient += -(2 / m) * x * (y - ((weight * x) + bias))
        bias_gradient += -(2 / m) * (y - ((weight * x) + bias))
    
    # Update weight and bias based on the gradients and learning rate
    weight -= (learn_rate * weight_gradient)
    bias -= (learn_rate * bias_gradient)

    return weight, bias

class LinearRegressionNumpy(ModelML):
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int):
        """
        Initializes the Linear Regression model using manual gradient descent

        Parameters:
        learn_rate: The learning rate for the gradient descent
        number_of_epochs: The number of training iterations to run
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Trains the linear regression model on the input data using gradient descent

        Parameters:
        features: The input features for training 
        labels: The target labels corresponding to the input features 
        """
        features = features.squeeze()
        self.weight = 0  # Initialize weight to 0
        self.bias = 0    # Initialize bias to 0

        # Perform gradient descent over the specified number of epochs
        for _ in range(self.number_of_epochs):
            self.cost = cost_function(features, labels, self.weight, self.bias)
            self.weight, self.bias = gradient_descent(features, labels, self.weight, self.bias, self.learn_rate)
        
        # Print the final state of the model
        print("Epoch: {}/{} Weight: {:.5f}, Bias: {:.5f} Cost: {:.5f}".format(
               self.number_of_epochs, self.number_of_epochs, self.weight, self.bias, self.cost))
    
    def __str__(self) -> str:
        return "Linear Regression (Numpy)"
