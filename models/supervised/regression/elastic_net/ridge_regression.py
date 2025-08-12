from typing import Tuple
import numpy as np

def cost_function(features: np.ndarray,
                  targets: np.ndarray,
                  weights: float,
                  bias: float,
                  reg_rate: float) -> float:
    """
    Computes the mean squared error cost for Ridge regression with L2 regularization

    Parameters:
        features: The input feature values
        targets: The target values corresponding to the input features 
        weights: The current weight value of the model
        bias: The current bias value of the model
        reg_rate: The regularization rate (lambda) for L2 penalty

    Returns:
        avg_cost: The average cost (MSE + L2 penalty) for the current weight and bias
    """
    predictions = np.dot(features, weights) + bias
    errors = targets - predictions
    mse = (errors ** 2).mean()
    l2_penalty = reg_rate * np.sum(weights ** 2)
    return mse + l2_penalty


def gradient_descent(features: np.ndarray,
                     targets: np.ndarray,
                     weights: float,
                     bias: float,
                     learn_rate: float,
                     reg_rate: float) -> Tuple[float, float]:
    """
    Performs one step of gradient descent for Ridge regression (L2 regularization)

    Parameters:
        features: The input feature values 
        targets: The target values corresponding to the input features 
        weights: The current weight value of the model
        bias: The current bias value of the model
        learn_rate: The learning rate for gradient descent
        reg_rate: The regularization rate (lambda) for L2 penalty

    Returns:
        weight: The updated weight value after one step of gradient descent
        bias: The updated bias value after one step of gradient descent
    """
    m = features.shape[0]
    predictions = np.dot(features, weights) + bias
    errors = targets - predictions

    # Compute gradients
    weights_grad = -(2 / m) * np.dot(features.T, errors) + 2 * reg_rate * weights
    bias_grad = -(2 / m) * np.sum(errors)

    # Update parameters
    weights -= learn_rate * weights_grad
    bias -= learn_rate * bias_grad

    return weights, bias

class RidgeRegression():
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 reg_rate: float) -> None:
        """
        Initializes the Ridge Regression model using manual gradient descent

        Parameters:
            learn_rate: The learning rate for gradient descent
            number_of_epochs: The number of training iterations to run
            reg_rate: The regularization rate (lambda) for L2 penalty
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.reg_rate = reg_rate

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Trains the Ridge regression model on the input data using gradient descent

        Parameters:
            features: The input features for training 
            targets: The target values corresponding to the input features 
        """
        _, n = features.shape
        self.weights = np.zeros(n)  # Initialize weight vector
        self.bias = 0.0             # Initialize bias

        for _ in range(self.number_of_epochs):
            self.cost = cost_function(features, targets,
                                      self.weights, self.bias,
                                      self.reg_rate)
            self.weights, self.bias = gradient_descent(
                features, targets,
                self.weights, self.bias,
                self.learn_rate, self.reg_rate
            )

    def predict(self, test_features: np.ndarray):
        """
        Predict continuous target values for given samples

        Parameters:
            test_features: Test feature matrix 

        Returns:
            np.ndarray: Predicted target values
        """
        predictions = test_features @ self.weights + self.bias
        
        return predictions

    def __str__(self) -> str:
        return "Ridge Regression"
