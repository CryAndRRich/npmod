import numpy as np
from .layers import Layer

class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = x.copy()
        return np.maximum(0.0, self.outputs)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = (self.outputs > 0.0).astype(np.float32)
        return previous_grad * grad_input
    
    def __str__(self) -> str:
        return "ReLU"


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01) -> None:
        """
        Parameters:
            alpha: Slope for the negative input values (Leaky value)
        """
        super().__init__()
        self.alpha = alpha
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = x.copy()
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = np.where(self.outputs > 0, 1, self.alpha)
        return previous_grad * grad_input
    
    def __str__(self) -> str:
        return "LeakyReLU"

class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = self.outputs * (1 - self.outputs)
        return previous_grad * grad_input
    
    def __str__(self) -> str:
        return "Sigmoid"

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
