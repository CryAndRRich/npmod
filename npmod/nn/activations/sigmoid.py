import numpy as np
from ..layers import Layer

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
    

class HardSigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.clip(0.2 * x + 0.5, 0.0, 1.0)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = np.where((self.outputs > 0) & (self.outputs < 1), 0.2, 0.0)
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "HardSigmoid"
    
    
class LogSigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.sigmoid_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return np.log(self.sigmoid_x + 1e-12)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = (1 - self.sigmoid_x)
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "LogSigmoid"