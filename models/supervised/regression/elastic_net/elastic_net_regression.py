import numpy as np
from typing import Tuple

def cost_function(features: np.ndarray,
                  targets: np.ndarray,
                  weights: np.ndarray,
                  bias: float,
                  alpha: float,
                  l1_ratio: float) -> float:
    """
    Compute ElasticNet cost function: MSE + L1 + L2 penalty

    Parameters:
        features: Feature matrix
        targets: Target vector
        weights: Weight vector
        bias: Bias term.
        alpha: Regularization strength.
        l1_ratio: Ratio between L1 and L2 penalty

    Returns:
        float: Computed cost value
    """
    preds = features @ weights + bias
    mse = np.mean((targets - preds) ** 2)
    l1_penalty = alpha * l1_ratio * np.sum(np.abs(weights))
    l2_penalty = alpha * (1 - l1_ratio) * np.sum(weights ** 2)
    return mse + l1_penalty + l2_penalty


def gradient_descent(features: np.ndarray,
                     targets: np.ndarray,
                     weights: np.ndarray,
                     bias: float,
                     learn_rate: float,
                     alpha: float,
                     l1_ratio: float) -> Tuple[np.ndarray, float]:
    """
    Perform one step of gradient descent for ElasticNet regression

    Parameters:
        features: Feature matrix
        targets: Target vector
        weights: Current weights
        bias: Current bias
        learn_rate: Learning rate
        alpha: Regularization strength
        l1_ratio: Ratio between L1 and L2 penalty

    Returns:
        Tuple[np.ndarray, float]: Updated weight vector and bias
    """
    m = features.shape[0]
    preds = features @ weights + bias
    errors = targets - preds

    w_grad = -(2 / m) * (features.T @ errors)
    b_grad = -(2 / m) * np.sum(errors)

    # Regularization terms
    w_grad += alpha * l1_ratio * np.sign(weights)               
    w_grad += 2 * alpha * (1 - l1_ratio) * weights            

    weights -= learn_rate * w_grad
    bias -= learn_rate * b_grad
    return weights, bias


class ElasticNetRegression:
    def __init__(self,
                 learn_rate: float = 0.01,
                 number_of_epochs: int = 1000,
                 alpha: float = 0.1,
                 l1_ratio: float = 0.5):
        """
        Parameters:
            learn_rate: Learning rate.
            number_of_epochs: Number of iterations for gradient descent.
            alpha: Regularization strength.
            l1_ratio: Ratio between L1 and L2 penalty.
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Train the ElasticNet regression model.

        Parameters:
            features: Feature matrix
            targets: Target vector
        """
        n_features = features.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.number_of_epochs):
            cost = cost_function(features, targets, self.weights, self.bias, self.alpha, self.l1_ratio)
            self.cost_history.append(cost)
            self.weights, self.bias = gradient_descent(features, targets, self.weights, self.bias,
                                              self.learn_rate, self.alpha, self.l1_ratio)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input features.

        Parameters:
            test_features: Feature matrix for prediction

        Returns:
            np.ndarray: Predicted target values.
        """
        predictions = test_features @ self.weights + self.bias
        return predictions
    
    def __str__(self) -> str:
            return "ElasticNet Regression"