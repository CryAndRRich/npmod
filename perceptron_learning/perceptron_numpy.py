from typing import List
import numpy as np
from base_model import ModelML

np.random.seed(42)

def cost_function(predictions: np.ndarray, 
                  labels: np.ndarray) -> float:
    """
    Computes the difference (error) between predictions and true labels

    Parameters:
    predictions: Predicted labels from the perceptron model
    labels: True target labels 

    --------------------------------------------------
    Returns:
    cost: The error between predictions and labels
    """
    cost = predictions - labels
    return cost

def heaviside_step(x_train: np.ndarray, 
                   weights: np.ndarray, 
                   bias: float) -> int|List[int]:
    """
    Applies the Heaviside step function to make a binary decision based on the weighted sum of inputs

    Parameters:
    x_train: The input feature values 
    weights: The current weight values 
    bias: The current bias value 

    --------------------------------------------------
    Returns:
    int|List[int]: A binary decision (0 or 1) after applying the step function
    """
    weighted_sum = x_train @ weights.T + bias
    
    try:
        return [int(weight >= 0) for weight in weighted_sum]
    except:
        return int(weighted_sum >= 0)

class PerceptronLearningNumpy(ModelML):
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int) -> None:
        """
        Initializes the Perceptron Learning model using the Perceptron Learning Algorithm

        Parameters:
        learn_rate: The learning rate for the model update
        number_of_epochs: The number of training iterations
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Trains the Perceptron model using the training data

        Parameters:
        features: Input feature matrix for training
        labels: True target labels corresponding to the input features
        """
        _, n = features.shape

        # Initialize weights randomly and set bias to 0
        self.weights = np.random.rand(n)
        self.bias = 0

        # Perform training over the specified number of epochs
        for _ in range(self.number_of_epochs):
            cost = 0
            for x, y in zip(features, labels):
                # Apply the Heaviside step function to make a prediction
                predictions = heaviside_step(x, self.weights, self.bias)
                
                # Compute the error (cost) based on the predictions and true labels
                error = cost_function(predictions, y)
                cost += error

                # Update the weights and bias using the Perceptron learning rule
                self.weights += self.learn_rate * error * x
                self.bias += self.learn_rate * error
            
    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray) -> None:
        """
        Predicts the labels for the test data using the trained Perceptron model

        Parameters:
        test_features: The input features for testing
        test_labels: The true labels corresponding to the test features
        """
        # Make predictions by applying the Heaviside step function
        predictions = 1 - np.array(heaviside_step(test_features, self.weights, self.bias))

        # Evaluate the predictions using accuracy and F1 score
        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
               self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Perceptron Learning model
        """
        return "Perceptron Learning Algorithm (Numpy)"
