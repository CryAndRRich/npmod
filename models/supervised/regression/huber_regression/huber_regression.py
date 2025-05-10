from typing import Tuple
import numpy as np
from ....base import Model

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
                  weight: float,
                  bias: float,
                  delta: float) -> float:
    """
    Compute average Huber loss cost for a linear model

    Parameters:
        features: Input feature values
        targets: True target values
        weight: Model weight
        bias: Model bias 
        delta: Huber threshold

    Returns:
        float: Mean Huber loss over all samples
    """
    m = features.shape[0]
    total_loss = 0.0
    for i in range(m):
        r = targets[i] - (weight * features[i] + bias)
        total_loss += huber_loss(r, delta)
    return total_loss / m

def gradient_descent(features: np.ndarray,
                     targets: np.ndarray,
                     weight: float,
                     bias: float,
                     learn_rate: float,
                     delta: float) -> Tuple[float, float]:
    """
    Perform one step of gradient descent using Huber loss

    Parameters:
        features: Input feature values
        targets: True target values
        weight: Current model weight
        bias: Current model bias
        learn_rate: Learning rate
        delta: Huber threshold

    Returns:
        Tuple[float, float]: Updated weight and bias
    """
    m = features.shape[0]
    weight_grad = 0.0
    bias_grad = 0.0
    for i in range(m):
        x = features[i]
        y = targets[i]
        y_pred = weight * x + bias
        r = y - y_pred
        dL_dr = huber_gradient(r, delta)
        # Chain rule: dL/dw = -dL/dr * x, dL/db = -dL/dr
        weight_grad += -(1/m) * dL_dr * x
        bias_grad   += -(1/m) * dL_dr
    weight -= learn_rate * weight_grad
    bias   -= learn_rate * bias_grad
    return weight, bias

class HuberRegression(Model):
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
        x = features.squeeze()
        self.weight = 0.0
        self.bias = 0.0
        for _ in range(self.number_of_epochs):
            self.cost = cost_function(x, targets, self.weight, self.bias, self.delta)
            self.weight, self.bias = gradient_descent(
                x, targets, self.weight, self.bias,
                self.learn_rate, self.delta
            )

    def predict(self,
                test_features: np.ndarray,
                test_targets: np.ndarray = None) -> np.ndarray:
        """
        Predict using the trained Huber regression model

        Parameters:
            test_features: Test feature array
            test_targets: Optional true targets for evaluation

        Returns:
            np.ndarray: Predicted target values
        """
        predictions = self.weight * test_features.squeeze() + self.bias
        
        if test_targets is not None:
            mse, r2 = self.regression_evaluate(predictions, test_targets)
            print("MSE: {:.5f} R-squared: {:.5f}".format(mse, r2))

        return predictions

    def __str__(self) -> str:
        return "Huber Regression"
