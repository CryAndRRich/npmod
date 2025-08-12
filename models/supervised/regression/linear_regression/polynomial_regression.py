from typing import Tuple
from itertools import combinations_with_replacement
import numpy as np

def polynomial_features(features: np.ndarray, 
                        degree: int) -> np.ndarray:
    """
    Generate polynomial features up to a given degree for each feature independently

    Parameters:
        features: Feature matrix of shape
        degree: Maximum polynomial degree

    Returns:
        np.ndarray: Expanded feature matrix with polynomial terms
    """
    n_samples, n_features = features.shape
    poly_X = []
    
    for deg in range(1, degree + 1):
        for items in combinations_with_replacement(range(n_features), deg):
            term = np.prod(features[:, items], axis=1)
            poly_X.append(term)
    
    return np.vstack(poly_X).T

def cost_function(features: np.ndarray, 
                  targets: np.ndarray, 
                  weights: np.ndarray, 
                  bias: float) -> float:
    """
    Computes the mean squared error cost for linear regression
    """
    predictions = np.dot(features, weights) + bias
    errors = targets - predictions
    mse = (errors ** 2).mean()
    return mse

def gradient_descent(features: np.ndarray, 
                     targets: np.ndarray, 
                     weights: np.ndarray, 
                     bias: float, 
                     learn_rate: float) -> Tuple[np.ndarray, float]:
    """
    Performs one step of gradient descent to update the weight and bias.
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

class PolynomialRegression:
    def __init__(self, 
                 degree: int, 
                 learn_rate: float = 0.01, 
                 number_of_epochs: int = 1000) -> None:
        """
        Initializes the Polynomial Regression model using manual gradient descent

        Parameters:
            degree: The maximum degree of the polynomial features
            learn_rate: The learning rate for the gradient descent
            number_of_epochs: The number of training iterations to run
        """
        self.degree = degree
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Trains the polynomial regression model on the input data using gradient descent
        """
        # Transform original features into polynomial features
        poly_X = polynomial_features(features, self.degree)

        _, n = poly_X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        # Gradient descent loop
        for _ in range(self.number_of_epochs):
            self.cost = cost_function(poly_X, targets, self.weights, self.bias)
            self.weights, self.bias = gradient_descent(
                poly_X, targets, self.weights, self.bias, self.learn_rate
            )

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for given samples
        """
        poly_X = polynomial_features(test_features, self.degree)
        predictions = np.dot(poly_X, self.weights) + self.bias
        return predictions

    def __str__(self) -> str:
        return "Polynomial Regression"
