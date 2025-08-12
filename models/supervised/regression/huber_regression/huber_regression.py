from typing import Tuple
import numpy as np

def huber_loss(r: float, delta: float) -> float:
    """
    Compute the Huber loss for a single residual

    Parameters:
        r: Residual (y - y_pred)
        delta: Threshold at which loss transitions from quadratic to linear

    Returns:
        float: Huber loss value
    """
    if abs(r) <= delta:
        return 0.5 * r ** 2
    return delta * (abs(r) - 0.5 * delta)

def huber_gradient(r: float, delta: float) -> float:
    """
    Compute the derivative of Huber loss with respect to the residual

    Parameters:
        r: Residual (y - y_pred)
        delta: Threshold parameter

    Returns:
        float: dL/dr value
    """
    if abs(r) <= delta:
        return r
    return delta * np.sign(r)

def cost_function(features: np.ndarray,
                  targets: np.ndarray,
                  weights: float,
                  bias: float,
                  delta: float) -> float:
    """
    Compute average Huber loss cost for a linear model

    Parameters:
        features: Input feature values
        targets: True target values
        weights: Model weight
        bias: Model bias 
        delta: Huber threshold

    Returns:
        float: Mean Huber loss over all samples
    """
    m = features.shape[0]
    predictions = features @ weights + bias
    residuals = targets - predictions
    total_loss = 0.0
    for r in residuals:
        total_loss += huber_loss(r, delta)
    return total_loss / m

def gradient_descent(features: np.ndarray,
                     targets: np.ndarray,
                     weights: float,
                     bias: float,
                     learn_rate: float,
                     delta: float) -> Tuple[float, float]:
    """
    Perform one step of gradient descent using Huber loss

    Parameters:
        features: Input feature values
        targets: True target values
        weights: Current model weight
        bias: Current model bias
        learn_rate: Learning rate
        delta: Huber threshold

    Returns:
        Tuple[float, float]: Updated weight and bias
    """
    m, n = features.shape
    predictions = features @ weights + bias
    residuals = targets - predictions

    weight_grad = np.zeros(n)
    bias_grad = 0.0

    for i in range(m):
        r = residuals[i]
        dL_dr = huber_gradient(r, delta)
        weight_grad += -(1 / m) * dL_dr * features[i]
        bias_grad   += -(1 / m) * dL_dr

    weights -= learn_rate * weight_grad
    bias -= learn_rate * bias_grad
    return weights, bias

class HuberRegression():
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 delta: float = 1.0) -> None:
        """
        Initialize Huber Regression using Huber loss to reduce outlier influence

        Parameters:
            learn_rate: Learning rate for gradient descent
            number_of_epochs: Number of training iterations
            delta: Threshold where loss changes from quadratic to linear
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.delta = delta

    def fit(self,
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Train the Huber regression model on the provided data

        Parameters:
            features: Training feature array
            targets: Training targets
        """
        _, n = features.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        for _ in range(self.number_of_epochs):
            self.cost = cost_function(features, targets, self.weights, self.bias, self.delta)
            self.weights, self.bias = gradient_descent(
                features, targets, self.weights, self.bias,
                self.learn_rate, self.delta
            )

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict using the trained Huber regression model

        Parameters:
            test_features: Test feature array

        Returns:
            np.ndarray: Predicted target values
        """
        predictions = test_features @ self.weights + self.bias

        return predictions

    def __str__(self) -> str:
        return "Huber Regression"
