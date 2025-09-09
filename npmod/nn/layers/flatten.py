import numpy as np
from ..layers import Layer

class Flatten(Layer):
    """
    Flatten layer to reshape input tensors
    """
    def __init__(self, keep_channels: bool = False) ->None:
        """
        Parameters:
            keep_channels: If True, retains the channel dimension during flattening.\
                           Otherwise, flattens all dimensions except the batch size
        """
        super().__init__()
        self.keep_channels = keep_channels
        self.forward_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.forward_shape = x.shape
        if self.keep_channels:
            return x.reshape(x.shape[0], x.shape[1], -1)
        else:
            return x.reshape(x.shape[0], -1)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        assert self.forward_shape is not None, "Must call forward before backward"
        return previous_grad.reshape(self.forward_shape)