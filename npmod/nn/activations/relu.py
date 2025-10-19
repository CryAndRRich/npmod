import numpy as np
from ..layers import Layer

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


class PReLU(Layer):
    def __init__(self, alpha: float = 0.25) -> None:
        """
        Parameters:
            alpha: Learnable slope for the negative input values
        """
        super().__init__()
        self.alpha = alpha
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = x.copy()
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = np.where(self.outputs > 0, 1.0, self.alpha)
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "PReLU"
    

class ELU(Layer):
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Parameters:
            alpha: scaling factor for negative region
        """
        super().__init__()
        self.alpha = alpha
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = x.copy()
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1.0))

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = np.where(self.outputs > 0, 1.0, self.alpha * np.exp(self.outputs))
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "ELU"


class SELU(Layer):
    def __init__(self) -> None:
        super().__init__()
        # Constants for SELU (from the original paper)
        self.alpha = 1.6732632423543772
        self.scale = 1.0507009873554805
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = x.copy()
        return self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1.0))

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = np.where(
            self.outputs > 0,
            self.scale,
            self.scale * self.alpha * np.exp(self.outputs),
        )
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "SELU"


class ReLU6(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = np.clip(x, 0.0, 6.0)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = ((self.outputs > 0.0) & (self.outputs < 6.0)).astype(np.float32)
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "ReLU6"