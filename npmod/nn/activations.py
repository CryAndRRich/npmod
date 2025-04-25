import numpy as np
from .layers import Layer

class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the ReLU activation function.
        Computes the element-wise maximum of 0 and the input

        Parameters:
            x: Input data

        Returns:
            np.ndarray: Output after applying ReLU activation
        """
        self.outputs = x.copy()
        return np.maximum(0.0, self.outputs)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the ReLU activation function.
        Computes the gradient of the loss with respect to the input

        Parameters:
            previous_grad: Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        grad_input = (self.outputs > 0.0).astype(np.float32)
        return previous_grad * grad_input


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01) -> None:
        """
        Initializes the LeakyReLU activation function

        Parameters:
            alpha: Slope for the negative input values (Leaky value)
        """
        super().__init__()
        self.alpha = alpha
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the LeakyReLU activation function.
        Computes the element-wise activation where negative inputs are scaled by alpha.

        Parameters:
            x: Input data

        Returns:
            np.ndarray: Output after applying LeakyReLU activation
        """
        self.outputs = x.copy()
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the LeakyReLU activation function.
        Computes the gradient of the loss with respect to the input

        Parameters:
            previous_grad: Gradient of the loss with respect to the output

        Returns:
            np.ndarray: Gradient of the loss with respect to the input
        """
        grad_input = np.where(self.outputs > 0, 1, self.alpha)
        return previous_grad * grad_input

class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the Sigmoid activation function.
        Computes the element-wise sigmoid of the input

        Parameters:
            x: Input data

        Returns:
            np.ndarray: Output after applying Sigmoid activation
        """
        exp_neg_x = np.exp(-x)
        self.outputs = np.divide(1, 1 + exp_neg_x)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the Sigmoid activation function.
        Computes the gradient of the loss with respect to the input

        Parameters:
            previous_grad: Gradient of the loss with respect to the output

        Returns:
            np.ndarray: Gradient of the loss with respect to the input
        """
        grad_input = self.outputs * (1 - self.outputs)
        return previous_grad * grad_input

class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the Tanh activation function.
        Computes the hyperbolic tangent of the input

        Parameters:
            x: Input data.

        Returns:
            np.ndarray: Output after applying Tanh activation
        """
        self.outputs = np.tanh(x)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the Tanh activation function.
        Computes the gradient of the loss with respect to the input

        Parameters:
            previous_grad: Gradient of the loss with respect to the output

        Returns:
            np.ndarray: Gradient of the loss with respect to the input
        """
        grad_input = 1 - np.square(self.outputs)
        return previous_grad * grad_input

class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the Softmax activation function.
        Computes the softmax of the input array along the last axis

        Parameters:
            x: Input data

        Returns:
            np.ndarray: Output after applying Softmax activation
        """
        # Subtract the maximum value for numerical stability
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.outputs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.outputs

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the Softmax activation function.
        Computes the gradient of the loss with respect to the input.
        This implementation assumes that the Softmax function is used
        in combination with a cross-entropy loss, where the gradient is simplified

        Parameters:
            previous_grad: Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        # Assuming the loss gradient is already computed using cross-entropy
        return previous_grad
