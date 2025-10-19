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
        return []

    def gradients(self):
        """
        Returns the gradients of the parameters
        """
        return []

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

from .flatten import Flatten
from .dropout import Dropout, DropConnect
from .embedding import Embedding
from .linear import Linear
from .conv import Conv
from .pooling import Pooling
from .batchnorm import BatchNorm, GroupNorm
from .weightnorm import WeightNorm
from .rnn import *