import numpy as np
from ..layers import Layer

class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract the maximum value for numerical stability
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.outputs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        # Vectorized softmax Jacobian multiplication
        dot = np.sum(previous_grad * self.outputs, axis=1, keepdims=True)
        grad_input = self.outputs * (previous_grad - dot)
        return grad_input
    
    def __str__(self) -> str:   
        return "Softmax"


class LogSoftmax(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(shifted_x), axis=-1, keepdims=True))
        self.outputs = shifted_x - logsumexp
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        softmax = np.exp(self.outputs)
        grad_input = previous_grad - np.sum(previous_grad, axis=1, keepdims=True) * softmax
        return grad_input

    def __str__(self) -> str:
        return "LogSoftmax"