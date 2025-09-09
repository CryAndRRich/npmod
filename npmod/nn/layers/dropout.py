import numpy as np
from ..layers import Layer

class Dropout(Layer):
    """
    Dropout layer for regularization
    """
    def __init__(self, keep_prob: float) ->None:
        """
        Parameters:
            keep_prob: Probability with which to keep the inputs
        """
        super().__init__()
        assert 0 < keep_prob <= 1, "Probability must be between 0 and 1"
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.binomial(1, self.keep_prob, x.shape).astype(np.float32)
            return (x * self.mask) / self.keep_prob
        else:
            return x

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        if self.training:
            return (previous_grad * self.mask) / self.keep_prob
        else:
            return previous_grad