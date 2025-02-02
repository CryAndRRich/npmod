from typing import Tuple
import numpy as np
from ..base_model import ModelML

np.random.seed(42)

def softmax_function(z: np.ndarray) -> np.ndarray:
    """
    Computes the softmax function for the given input scores

    --------------------------------------------------
    Parameters:
        z: The input score matrix (before softmax)

    --------------------------------------------------
    Returns:
        np.ndarray: The output probability distribution after applying softmax
    """
    exp_z = np.exp(z - np.max(z))  # Numerical stability by subtracting max(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(labels: np.ndarray, 
                  probs: np.ndarray, 
                  number_of_samples: int) -> float:
    """
    Computes the cross-entropy loss between the true labels and predicted probabilities

    --------------------------------------------------
    Parameters:
        labels: One-hot encoded true labels
        probs: Predicted probabilities from the softmax function
        number_of_samples: The number of training samples

    --------------------------------------------------
    Returns:
        cost: The cross-entropy loss
    """
    cost = - np.sum(labels * np.log(probs)) / number_of_samples
    return cost

def one_hot_encode(labels: np.ndarray, 
                   number_of_samples: int, 
                   number_of_classes: int) -> np.ndarray:
    """
    Converts labels into a one-hot encoded format

    --------------------------------------------------
    Parameters:
        labels: The original integer labels
        number_of_samples: The total number of samples
        number_of_classes: The number of unique classes in the dataset

    --------------------------------------------------
    Returns:
        one_hot: The one-hot encoded labels
    """
    one_hot = np.zeros((number_of_samples, number_of_classes))
    one_hot[np.arange(number_of_samples), labels.T] = 1
    return one_hot

def gradient_descent(features: np.ndarray, 
                     labels: np.ndarray, 
                     probs: np.ndarray, 
                     weights: np.ndarray, 
                     bias: np.ndarray, 
                     learn_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs one step of gradient descent to update the model's weights and bias

    --------------------------------------------------
    Parameters:
        features: The input features matrix
        labels: One-hot encoded true labels
        probs: Predicted probabilities from the softmax function
        weights: The current weight matrix
        bias: The current bias values
        learn_rate: The learning rate for gradient descent

    --------------------------------------------------
    Returns:
        weights: The updated weight matrix
        bias: The updated bias values
    """
    m = features.shape[0]

    # Compute gradients for weights and bias
    weights_gradient = np.dot(features.T, (probs - labels)) / m
    bias_gradient = np.sum(probs - labels, axis=0) / m
    
    # Update weights and bias using the computed gradients
    weights -= learn_rate * weights_gradient.T
    bias -= learn_rate * bias_gradient

    return weights, bias

class SoftmaxRegressionNumpy(ModelML):
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int, 
                 number_of_classes: int = 2) -> None:
        """
        Initializes the Softmax Regression model using gradient descent

        --------------------------------------------------
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
            labels: np.ndarray) -> None:
        """
        Trains the Softmax Regression model on the input data

        --------------------------------------------------
        Parameters:
            features: Input feature matrix for training
            labels: True target labels corresponding to the input features
        """
        m, n = features.shape

        # Initialize weights randomly and bias to zeros
        self.weights = np.random.rand(self.number_of_classes, n)
        self.bias = np.zeros((1, self.number_of_classes))

        # Perform training over the specified number of epochs
        for _ in range(self.number_of_epochs):
            # Convert labels to one-hot encoding
            y_one_hot = one_hot_encode(labels, m, self.number_of_classes)

            # Compute scores (logits) and predicted probabilities using softmax
            scores = np.dot(features, self.weights.T) + self.bias
            probs = softmax_function(scores)

            # Compute the cross-entropy loss
            cost = cross_entropy(y_one_hot, probs, m)

            # Update weights and bias using gradient descent
            self.weights, self.bias = gradient_descent(features, y_one_hot, probs, self.weights, self.bias, self.learn_rate)
    
    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray,
                get_accuracy: bool = True) -> np.ndarray:
        """
        Predicts the labels for the test data using the trained Softmax Regression model

        --------------------------------------------------
        Parameters:
            test_features: The input features for testing
            test_labels: The true labels corresponding to the test features
            get_accuracy: If True, calculates and prints the accuracy of predictions

        --------------------------------------------------
        Returns:
            predictions: The prediction labels
        """
        # Compute scores (logits) and predicted probabilities using softmax
        scores = np.dot(test_features, self.weights.T) + self.bias
        probs = softmax_function(scores)

        # Predict the class label with the highest probability
        predictions = np.argmax(probs, axis=1)[:, np.newaxis]

        if get_accuracy:
            # Evaluate the predictions using accuracy and F1 score
            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                self.number_of_epochs, self.number_of_epochs, accuracy, f1))
        
        return predictions
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Softmax Regression model
        """
        return "Softmax Regression (Numpy)"
