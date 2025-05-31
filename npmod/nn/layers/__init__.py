import numpy as np

class Layer():
    """
    Fundamental building block of the model
    """
    def __init__(self) -> None:
        self.training = True

    def parameters(self):
        """
        Returns the parameters of the layer
        """
        return
        yield

    def gradients(self):
        """
        Returns the gradients of the parameters
        """
        return
        yield

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer
        """
        pass

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer
        """
        pass
    
    def __call__(self, x):
        return self.forward(x)

from .linear import Linear
from .dropout import Dropout
from .flatten import Flatten
from .embedding import Embedding
from .batchnorm import BatchNorm

from .conv import *
from .rnn import *