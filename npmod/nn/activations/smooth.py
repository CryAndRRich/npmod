import numpy as np
from ..layers import Layer

class GELU(Layer):
    def __init__(self, approximate: bool = True) -> None:
        """
        Parameters:
            approximate: whether to use tanh-based fast approximation
        """
        super().__init__()
        self.approximate = approximate
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs = x.copy()
        if self.approximate:
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        else:
            return 0.5 * x * (1.0 + np.erf(x / np.sqrt(2)))

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        x = self.outputs

        if self.approximate:
            # Fast tanh-based derivative
            sqrt_2_over_pi = np.sqrt(2 / np.pi)
            inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
            tanh_inner = np.tanh(inner)
            sech2_inner = 1 - tanh_inner**2
            grad_gelu = (
                0.5 * (1.0 + tanh_inner)
                + 0.5 * x * sech2_inner * sqrt_2_over_pi * (1 + 3 * 0.044715 * x ** 2)
            )
        else:
            # Exact GELU derivative
            erf_term = np.erf(x / np.sqrt(2))
            pdf_term = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
            grad_gelu = 0.5 * (1.0 + erf_term) + x * pdf_term

        return previous_grad * grad_gelu

    def __str__(self) -> str:
        return "GELU"


class Swish(Layer):
    def __init__(self, beta: float = 1.0) -> None:
        """
        Parameters:
            beta: scaling factor for the input
        """
        super().__init__()
        self.beta = beta
        self.sigmoid_x = None
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.beta * x
        self.sigmoid_x = 1 / (1 + np.exp(-z))
        self.outputs = x * self.sigmoid_x
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_swish = self.sigmoid_x + self.beta * self.outputs * (1 - self.sigmoid_x)
        return previous_grad * grad_swish

    def __str__(self) -> str:
        return "Swish"


class HardSwish(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None
        self.hsig = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x.copy()
        self.hsig = np.clip(x / 6 + 0.5, 0.0, 1.0)
        self.outputs = x * self.hsig
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = self.hsig + (self.x / 6.0) * ((self.x > -3) & (self.x < 3))
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "HardSwish"
    

class Softplus(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x.copy()
        self.outputs = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = 1 / (1 + np.exp(-self.x))
        return previous_grad * grad_input

    def __str__(self) -> str:
        return "Softplus"


class Mish(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.softplus_x = None
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x.copy()
        self.softplus_x = np.log1p(np.exp(x))
        self.outputs = x * np.tanh(self.softplus_x)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        sp = self.softplus_x
        tanh_sp = np.tanh(sp)
        sigmoid_x = 1 / (1 + np.exp(-self.x))
        grad_mish = tanh_sp + self.x * sigmoid_x * (1 - tanh_sp ** 2)
        return previous_grad * grad_mish

    def __str__(self) -> str:
        return "Mish"
