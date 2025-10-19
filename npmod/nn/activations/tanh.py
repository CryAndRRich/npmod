import numpy as np
from ..layers import Layer

class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.tanh(x)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = 1 - np.square(self.outputs)
        return previous_grad * grad_input
    
    def __str__(self) -> str:   
        return "Tanh"


class HardTanh(Layer):
    def __init__(self, 
                 min_val: float = -1.0, 
                 max_val: float = 1.0) -> None:
        """
        Parameters:
            min_val: minimum clipping value
            max_val: maximum clipping value
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.clip(x, self.min_val, self.max_val)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = ((self.outputs > self.min_val) & (self.outputs < self.max_val)).astype(np.float32)
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "HardTanh"