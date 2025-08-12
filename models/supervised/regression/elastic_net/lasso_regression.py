from typing import Tuple
import numpy as np

def cost_function(features: np.ndarray,
                  targets: np.ndarray,
                  weights: float,
                  bias: float,
                  reg_rate: float) -> float:
    """
    Computes the mean squared error cost for Lasso regression with L1 regularization

    Parameters:
        features: The input feature values
        targets: The target values corresponding to the input features 
        weights: The current weight value of the model
        bias: The current bias value of the model
        reg_rate: The regularization rate (lambda) for L1 penalty

    Returns:
        avg_cost: The average cost (MSE + L1 penalty) for the current weight and bias
    """
    predictions = features @ weights + bias
    errors = targets - predictions
    mse = np.mean(errors ** 2)
    l1_penalty = reg_rate * np.sum(np.abs(weights))
    return mse + l1_penalty

def gradient_descent(features: np.ndarray,
                     targets: np.ndarray,
                     weights: float,
                     bias: float,
                     learn_rate: float,
                     reg_rate: float) -> Tuple[float, float]:
    """
    Performs one step of gradient descent for Lasso regression (L1 regularization)

    Parameters:
        features: The input feature values 
        targets: The target values corresponding to the input features 
        weights: The current weight value of the model
        bias: The current bias value of the model
        learn_rate: The learning rate for gradient descent
        reg_rate: The regularization rate (lambda) for L1 penalty

    Returns:
        weight: The updated weight value after one step of gradient descent
        bias: The updated bias value after one step of gradient descent
    """
    m = features.shape[0]
    predictions = features @ weights + bias
    errors = targets - predictions

    # Gradients
    weight_grad = -(2 / m) * (features.T @ errors)
    bias_grad = -(2 / m) * np.sum(errors)

    # Add sub-gradient for L1 regularization
    weight_grad += reg_rate * np.sign(weights)

    weights -= learn_rate * weight_grad
    bias -= learn_rate * bias_grad

    return weights, bias

class LassoRegression():
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 reg_rate: float) -> None:
        """
        Initializes the Lasso Regression model using manual gradient descent

        Parameters:
            learn_rate: The learning rate for gradient descent
            number_of_epochs: The number of training iterations to run
            reg_rate: The regularization rate (lambda) for L1 penalty
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.reg_rate = reg_rate

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Trains the Lasso regression model on the input data using gradient descent

        Parameters:
            features: The input features for training 
            targets: The target values corresponding to the input features 
        """
        _, n = features.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for _ in range(self.number_of_epochs):
            self.cost = cost_function(features, targets, self.weights, self.bias, self.reg_rate)
            self.weights, self.bias = gradient_descent(
                features, targets, self.weights, self.bias,
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
        return "Lasso Regression"
