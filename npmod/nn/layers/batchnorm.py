import numpy as np
from ..layers import Layer

class BatchNorm(Layer):
    """
    Batch Normalization layer
    """
    def __init__(self, 
                 num_features: int, 
                 eps: float = 1e-5) -> None:
        """
        Parameters:
            num_features: Number of features (channels)
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.eps = eps
        self.num_features = num_features

        # Learnable parameters
        self.gamma = np.ones((num_features,), dtype=np.float32)
        self.beta = np.zeros((num_features,), dtype=np.float32)

        # Gradients
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)

        # Cache
        self.input = None
        self.normalized_input = None
        self.batch_mean = None
        self.batch_var = None

    def parameters(self):
        yield self.gamma
        yield self.beta

    def gradients(self):
        yield self.gamma_grad
        yield self.beta_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        # Compute batch statistics
        self.batch_mean = np.mean(x, axis=0, keepdims=True)
        self.batch_var = np.var(x, axis=0, keepdims=True)
        self.normalized_input = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
        out = self.normalized_input * self.gamma + self.beta
        return out

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        N, _ = previous_grad.shape

        self.gamma_grad = np.sum(previous_grad * self.normalized_input, axis=0)
        self.beta_grad = np.sum(previous_grad, axis=0)

        x_mu = self.input - self.batch_mean
        std_inv = 1.0 / np.sqrt(self.batch_var + self.eps)

        grad_norm = previous_grad * self.gamma
        grad_input = (1. / N) * std_inv * (
            N * grad_norm - np.sum(grad_norm, axis=0) - x_mu * np.sum(grad_norm * x_mu, axis=0) * (std_inv ** 2)
        )
        return grad_input