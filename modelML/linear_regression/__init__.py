import numpy as np
import torch
from .linear_regression_numpy import LinearRegressionNumpy
from .linear_regression_pytorch import LinearRegressionPytorch

class LinearRegression:
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int, 
                 type: str = "numpy") -> None:
        """
        Initializes the Linear Regression model by selecting the appropriate implementation type

        --------------------------------------------------
        Parameters:
            learn_rate: Learning rate for gradient descent optimization
            number_of_epochs: Number of iterations (epochs) for training
            type: Type of implementation ("numpy" or "pytorch")
        """
        if type == "numpy":
            self.inherit = LinearRegressionNumpy(learn_rate, number_of_epochs)
        elif type == "pytorch":
            self.inherit = LinearRegressionPytorch(learn_rate, number_of_epochs)
        else: 
            raise ValueError(f"Type must be 'numpy' or 'pytorch'")

    def fit(self, 
            features: np.ndarray | torch.Tensor, 
            labels: np.ndarray | torch.Tensor) -> None:
        
        self.inherit.fit(features, labels)

    def __str__(self) -> str:
        return self.inherit.__str__()
