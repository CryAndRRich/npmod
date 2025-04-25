from typing import List
import numpy as np
import torch
from .mlp_numpy import MLPNumpy
from .mlp_pytorch import MLPPytorch

class MultilayerPerceptron():
    def __init__(self,
                 batch_size: int,
                 learn_rate: float, 
                 number_of_epochs: int,
                 n_layers: int,
                 n_neurons: List[int], 
                 type: str = "numpy") -> None:
        """
        Initializes the MLP model by selecting the appropriate implementation type

        Parameters:
            batch_size: Size of a training mini-batch
            learn_rate: Learning rate for gradient descent optimization
            number_of_epochs: Number of iterations (epochs) for training
            n_layers: Number of hidden layers
            n_neurons: Number of neurons in each hidden layer
            type: Type of implementation ("numpy" or "pytorch")
        """
        if type == "numpy":
            self.inherit = MLPNumpy(batch_size, learn_rate, number_of_epochs, n_layers, n_neurons)
        elif type == "pytorch":
            self.inherit = MLPPytorch(batch_size, learn_rate, number_of_epochs, n_layers, n_neurons)
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
