from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class Model():
    """
    Base class for Machine Learning models

    This class provides a template for implementing machine learning models with 
    methods for fitting, prediction, evaluation, and string representation
    """
    def __init__(self):
        pass
    
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Fits the model to the training data

        --------------------------------------------------
        Parameters:
            features: The input features for training the model
            labels: The corresponding target labels for the training data
        """
        pass

    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray,
                get_accuracy: bool = True) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        --------------------------------------------------
        Parameters:
            test_features: The input features for testing
            test_labels: The true target labels corresponding to the test features
            get_accuracy: If True, calculates and prints the accuracy of predictions

        --------------------------------------------------
        Returns:
            predictions: The prediction labels
        """
        pass

    def evaluate(self, 
                 predictions: np.ndarray, 
                 test_labels: np.ndarray) -> Tuple[float, float]:
        """
        Evaluates the performance of the model using accuracy and F1-score

        --------------------------------------------------
        Parameters:
            predictions: The predicted labels by the model
            test_labels: The true labels for the test set

        --------------------------------------------------
        Returns:
            accuracy, f1: The accuracy and F1-score of the model
        """
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average="weighted", zero_division=0)

        return (accuracy, f1)
    
    def __str__(self):
        """
        Provides a string representation of the model
        """
        pass