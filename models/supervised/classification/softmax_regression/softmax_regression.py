from typing import Tuple
import numpy as np

np.random.seed(42)

def softmax_function(z: np.ndarray) -> np.ndarray:
    """
    Computes the softmax function for the given input scores

    Parameters:
        z: The input score matrix (before softmax)

    Returns:
        np.ndarray: The output probability distribution after applying softmax
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability by subtracting max(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(targets: np.ndarray, 
                  probs: np.ndarray, 
                  number_of_samples: int) -> float:
    """
    Computes the cross-entropy loss between the true targets and predicted probabilities

    Parameters:
        targets: One-hot encoded true targets
        probs: Predicted probabilities from the softmax function
        number_of_samples: The number of training samples

    Returns:
        cost: The cross-entropy loss
    """
    cost = - np.sum(targets * np.log(probs)) / number_of_samples
    return cost

def one_hot_encode(targets: np.ndarray, 
                   number_of_samples: int, 
                   number_of_classes: int) -> np.ndarray:
    """
    Converts targets into a one-hot encoded format

    Parameters:
        targets: The original integer targets
        number_of_samples: The total number of samples
        number_of_classes: The number of unique classes in the dataset

    Returns:
        one_hot: The one-hot encoded targets
    """
    one_hot = np.zeros((number_of_samples, number_of_classes))
    one_hot[np.arange(number_of_samples), targets.T] = 1
    return one_hot

def gradient_descent(features: np.ndarray, 
                     targets: np.ndarray, 
                     probs: np.ndarray, 
                     weights: np.ndarray, 
                     bias: np.ndarray, 
                     learn_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs one step of gradient descent to update the model's weights and bias

    Parameters:
        features: The input features matrix
        targets: One-hot encoded true targets
        probs: Predicted probabilities from the softmax function
        weights: The current weight matrix
        bias: The current bias values
        learn_rate: The learning rate for gradient descent

    Returns:
        weights: The updated weight matrix
        bias: The updated bias values
    """
    m = features.shape[0]

    # Compute gradients for weights and bias
    weights_gradient = np.dot(features.T, (probs - targets)) / m
    bias_gradient = np.sum(probs - targets, axis=0) / m
    
    # Update weights and bias using the computed gradients
    weights -= learn_rate * weights_gradient.T
    bias -= learn_rate * bias_gradient

    return weights, bias

class SoftmaxRegression():
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int, 
                 number_of_classes: int = 2) -> None:
        """
        Initializes the Softmax Regression model using gradient descent

        Parameters:
            learn_rate: The learning rate for gradient descent
            number_of_epochs: The number of training iterations (epochs)
            number_of_classes: The number of unique classes in the classification problem
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.number_of_classes = number_of_classes
    
    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Trains the Softmax Regression model on the input data

        Parameters:
            features: Input feature matrix for training
            targets: True target targets corresponding to the input features
        """
        m, n = features.shape

        # Initialize weights randomly and bias to zeros
        self.weights = np.random.rand(self.number_of_classes, n)
        self.bias = np.zeros((1, self.number_of_classes))

        # Perform training over the specified number of epochs
        for _ in range(self.number_of_epochs):
            # Convert targets to one-hot encoding
            y_one_hot = one_hot_encode(targets, m, self.number_of_classes)

            # Compute scores (logits) and predicted probabilities using softmax
            scores = np.dot(features, self.weights.T) + self.bias
            probs = softmax_function(scores)

            # Compute the cross-entropy loss
            # cost = cross_entropy(y_one_hot, probs, m)

            # Update weights and bias using gradient descent
            self.weights, self.bias = gradient_descent(features, y_one_hot, probs, self.weights, self.bias, self.learn_rate)
    
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predicts the targets for the test data using the trained Softmax Regression model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        # Compute scores (logits) and predicted probabilities using softmax
        scores = np.dot(test_features, self.weights.T) + self.bias
        probs = softmax_function(scores)

        # Predict the class target with the highest probability
        predictions = np.argmax(probs, axis=1)[:, np.newaxis]

        return predictions
    
    def __str__(self) -> str:
        return "Softmax Regression"
