import numpy as np
import torch
from .softmax_regression_numpy import SoftmaxRegressionNumpy
from .softmax_regression_pytorch import SoftmaxRegressionPytorch

class SoftmaxRegression():
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int, 
                 number_of_classes: int = 2, 
                 type: str = "numpy"):
        """
        Initializes the Softmax Regression model by selecting the desired implementation 

        Parameters:
            learn_rate: The learning rate for the optimizer (controls the rate of weight updates)
            number_of_epochs: The number of training iterations (epochs)
            number_of_classes: The number of output classes (default is 2 for binary classification)
            type: The implementation type ('numpy' for NumPy-based or 'pytorch' for PyTorch-based)
        """
        if type == "numpy":
            self.inherit = SoftmaxRegressionNumpy(learn_rate, number_of_epochs, number_of_classes)
        elif type == "pytorch":
            self.inherit = SoftmaxRegressionPytorch(learn_rate, number_of_epochs, number_of_classes)
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
