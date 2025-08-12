from typing import Tuple
import numpy as np

def cost_function(features: np.ndarray, 
                  targets: np.ndarray, 
                  weights: np.ndarray, 
                  bias: float) -> float:
    """
    Computes the mean squared error cost for linear regression

    Parameters:
        features: The input feature values 
        targets: The target targets corresponding to the input features 
        weights: The current weight value of the model
        bias: The current bias value of the model

    Returns:
        avg_cost: The average cost (mean squared error) for the current weight and bias
    """
    predictions = np.dot(features, weights) + bias
    errors = targets - predictions
    mse = (errors ** 2).mean()
    return mse

def gradient_descent(features: np.ndarray, 
                     targets: np.ndarray, 
                     weights: np.ndarray, 
                     bias: float, 
                     learn_rate: float) -> Tuple[float, float]:
    """
    Performs one step of gradient descent to update the weight and bias

    Parameters:
        features: The input feature values 
        targets: The target targets corresponding to the input features 
        weights: The current weight value of the model
        bias: The current bias value of the model
        learn_rate: The learning rate for gradient descent

    Returns:
        weight: The updated weight value after one step of gradient descent
        bias: The updated bias value after one step of gradient descent
    """
    m = features.shape[0]
    predictions = np.dot(features, weights) + bias
    errors = targets - predictions

    # Compute gradients
    weights_grad = -(2 / m) * np.dot(features.T, errors)
    bias_grad = -(2 / m) * np.sum(errors)

    # Update parameters
    weights -= learn_rate * weights_grad
    bias -= learn_rate * bias_grad

    return weights, bias

class LinearRegression():
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
            targets: The target corresponding to the input features 
        """
        _, n = features.shape
        self.weights = np.zeros(n)
        self.bias = 0

        # Perform gradient descent over the specified number of epochs
        for _ in range(self.number_of_epochs):
            self.cost = cost_function(features, targets, self.weights, self.bias)
            self.weights, self.bias = gradient_descent(
                features, targets, 
                self.weights, self.bias, self.learn_rate
            )
    
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for given samples

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        predictions = np.dot(test_features, self.weights) + self.bias

        return predictions
    
    def __str__(self) -> str:
        return "Linear Regression"
