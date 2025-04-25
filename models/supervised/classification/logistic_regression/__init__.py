import numpy as np
import torch
from .logistic_regression_numpy import LogisticRegressionNumpy
from .logistic_regression_pytorch import LogisticRegressionPytorch

class LogisticRegression:
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int, 
                 type: str = "numpy") -> None:
        """
        Initializes the Logistic Regression model by selecting the appropriate implementation type

        Parameters:
            learn_rate: Learning rate for gradient descent optimization
            number_of_epochs: Number of iterations (epochs) for training
            type: Type of implementation ("numpy" or "pytorch")
        """
        if type == "numpy":
            self.inherit = LogisticRegressionNumpy(learn_rate, number_of_epochs)
        elif type == "pytorch":
            self.inherit = LogisticRegressionPytorch(learn_rate, number_of_epochs)
        else: 
            raise ValueError(f"Type must be 'numpy' or 'pytorch'")

    def fit(self, 
            features: np.ndarray | torch.Tensor, 
            labels: np.ndarray | torch.Tensor) -> None:
        
        self.inherit.fit(features, labels)

    def predict(self, 
                test_features: np.ndarray | torch.Tensor, 
                test_labels: np.ndarray | torch.Tensor,
                get_accuracy: bool = True) -> np.ndarray | torch.Tensor:
        
        predictions = self.inherit.predict(test_features, test_labels, get_accuracy)
        return predictions

    def __str__(self) -> str:
        return self.inherit.__str__()
