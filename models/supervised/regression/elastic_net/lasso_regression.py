from typing import Tuple
import numpy as np

def cost_function(features: np.ndarray,
                  targets: np.ndarray,
                  weight: float,
                  bias: float,
                  reg_rate: float) -> float:
    """
    Computes the mean squared error cost for Lasso regression with L1 regularization

    Parameters:
        features: The input feature values
        targets: The target values corresponding to the input features 
        weight: The current weight value of the model
        bias: The current bias value of the model
        reg_rate: The regularization rate (lambda) for L1 penalty

    Returns:
        avg_cost: The average cost (MSE + L1 penalty) for the current weight and bias
    """
    m = features.shape[0]
    total_error = 0.0

    # Compute sum of squared errors
    for i in range(m):
        x = features[i]
        y = targets[i]
        total_error += (y - (weight * x + bias)) ** 2

    mse = total_error / m
    l1_penalty = reg_rate * abs(weight)

    avg_cost = mse + l1_penalty
    return avg_cost


def gradient_descent(features: np.ndarray,
                     targets: np.ndarray,
                     weight: float,
                     bias: float,
                     learn_rate: float,
                     reg_rate: float) -> Tuple[float, float]:
    """
    Performs one step of gradient descent for Lasso regression (L1 regularization)

    Parameters:
        features: The input feature values 
        targets: The target values corresponding to the input features 
        weight: The current weight value of the model
        bias: The current bias value of the model
        learn_rate: The learning rate for gradient descent
        reg_rate: The regularization rate (lambda) for L1 penalty

    Returns:
        weight: The updated weight value after one step of gradient descent
        bias: The updated bias value after one step of gradient descent
    """
    m = features.shape[0]
    weight_grad = 0.0
    bias_grad = 0.0

    # Compute gradients for weight and bias
    for i in range(m):
        x = features[i]
        y = targets[i]
        error = y - (weight * x + bias)
        weight_grad += -(2 / m) * x * error
        bias_grad   += -(2 / m) * error

    # Add subgradient of L1 penalty for weight
    if weight > 0:
        weight_grad += reg_rate
    elif weight < 0:
        weight_grad -= reg_rate
    # if weight == 0, subgradient can be anything between [-reg_rate, reg_rate]; we leave it

    # Update parameters
    weight -= learn_rate * weight_grad
    bias   -= learn_rate * bias_grad

    return weight, bias

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
        features = features.squeeze()
        self.weight = 0.0  # Initialize weight
        self.bias = 0.0    # Initialize bias

        for _ in range(1, self.number_of_epochs + 1):
            # Compute current cost with L1 penalty
            self.cost = cost_function(features, targets,
                                      self.weight, self.bias,
                                      self.reg_rate)
            # Update parameters via gradient descent
            self.weight, self.bias = gradient_descent(
                features, targets,
                self.weight, self.bias,
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
        predictions = (self.weight * test_features) + self.bias

        return predictions

    def __str__(self) -> str:
        return "Lasso Regression"
