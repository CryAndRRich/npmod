from typing import Tuple
import numpy as np
from ....base import Model

def cost_function(features: np.ndarray, 
                  targets: np.ndarray, 
                  weight: float, 
                  bias: float) -> float:
    """
    Computes the mean squared error cost for linear regression

    Parameters:
        features: The input feature values 
        targets: The target targets corresponding to the input features 
        weight: The current weight value of the model
        bias: The current bias value of the model

    Returns:
        avg_cost: The average cost (mean squared error) for the current weight and bias
    """
    m = features.shape[0]
    total_cost = 0

    # Compute the total squared error
    for i in range(m):
        x = features[i]
        y = targets[i]
        total_cost += (y - (weight * x + bias)) ** 2
    
    # Average cost over all data points
    avg_cost = total_cost / m
    return avg_cost

def gradient_descent(features: np.ndarray, 
                     targets: np.ndarray, 
                     weight: float, 
                     bias: float, 
                     learn_rate: float) -> Tuple[float, float]:
    """
    Performs one step of gradient descent to update the model's weight and bias

    Parameters:
        features: The input feature values 
        targets: The target targets corresponding to the input features 
        weight: The current weight value of the model
        bias: The current bias value of the model
        learn_rate: The learning rate for gradient descent

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
        y = targets[i]

        weight_gradient += -(2 / m) * x * (y - ((weight * x) + bias))
        bias_gradient += -(2 / m) * (y - ((weight * x) + bias))
    
    # Update weight and bias based on the gradients and learning rate
    weight -= (learn_rate * weight_gradient)
    bias -= (learn_rate * bias_gradient)

    return weight, bias

class LinearRegressionNumpy(Model):
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int) -> None:
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
            targets: np.ndarray) -> None:
        """
        Trains the linear regression model on the input data using gradient descent

        Parameters:
            features: The input features for training 
            targets: The target targets corresponding to the input features 
        """
        features = features.squeeze()
        self.weight = 0  # Initialize weight to 0
        self.bias = 0    # Initialize bias to 0

        # Perform gradient descent over the specified number of epochs
        for _ in range(self.number_of_epochs):
            self.cost = cost_function(features, targets, self.weight, self.bias)
            self.weight, self.bias = gradient_descent(features, targets, self.weight, self.bias, self.learn_rate)
    
    def predict(self, 
                test_features: np.ndarray, 
                test_targets: np.ndarray = None) -> np.ndarray:

        """
        Predict continuous target values for given samples

        Parameters:
            test_features: Test feature matrix
            test_targets: Test target values (optional, for evaluation)

        Returns:
            np.ndarray: Predicted target values
        """
        
        predictions = (self.weight * test_features) + self.bias

        if test_targets is not None:
            mse, r2 = self.regression_evaluate(predictions, test_targets)
            print("MSE: {:.5f} R-squared: {:.5f}".format(mse, r2))

        return predictions
    
    def __str__(self) -> str:
        return "Linear Regression (Numpy)"
