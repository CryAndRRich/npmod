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
        if x.ndim == 2:
            axis = (0,)
            reshape_shape = (1, -1)
        elif x.ndim == 4:
            axis = (0, 2, 3)
            reshape_shape = (1, -1, 1, 1)
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        # Compute mean and variance
        self.batch_mean = np.mean(x, axis=axis, keepdims=True)
        self.batch_var = np.var(x, axis=axis, keepdims=True)

        # Normalize
        self.normalized_input = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)

        # Apply learnable scale and shift
        out = self.normalized_input * self.gamma.reshape(reshape_shape) + self.beta.reshape(reshape_shape)
        return out

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        x = self.input
        N = x.shape[0]
        if x.ndim == 2:
            axis = (0,)
            reshape_shape = (1, -1)
            dims = N
        elif x.ndim == 4:
            axis = (0, 2, 3)
            reshape_shape = (1, -1, 1, 1)
            dims = N * x.shape[2] * x.shape[3]
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        # Gradients of gamma and beta
        self.gamma_grad = np.sum(previous_grad * self.normalized_input, axis=axis, keepdims=False)
        self.beta_grad = np.sum(previous_grad, axis=axis, keepdims=False)

        std_inv = 1.0 / np.sqrt(self.batch_var + self.eps)
        grad_norm = previous_grad * self.gamma.reshape(reshape_shape)

        grad_input = (1. / dims) * std_inv * (
            dims * grad_norm
            - np.sum(grad_norm, axis=axis, keepdims=True)
            - (x - self.batch_mean) * np.sum(grad_norm * (x - self.batch_mean), axis=axis, keepdims=True) * (std_inv ** 2)
        )
        return grad_input
    
class GroupNorm(Layer):
    """
    Group Normalization layer
    - When num_groups = 1, becomes InstanceNorm
    - When num_groups = num_features, becomes LayerNorm
    """
    def __init__(self, 
                 num_features: int, 
                 num_groups: int, 
                 eps: float = 1e-5) -> None:
        """
        Parameters:
            num_features: Number of features (channels)
            num_groups: Number of groups to divide the channels into
            eps: Small value to avoid division by zero
        """
        super().__init__()
        assert num_features % num_groups == 0, "num_features must be divisible by num_groups"
        self.num_features = num_features
        self.num_groups = num_groups
        self.group_size = num_features // num_groups
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones((num_features,), dtype=np.float32)
        self.beta = np.zeros((num_features,), dtype=np.float32)

        # Gradients
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)

        self.x = None
        self.normalized = None
        self.mean = None
        self.var = None

    def parameters(self):
        yield self.gamma
        yield self.beta

    def gradients(self):
        yield self.gamma_grad
        yield self.beta_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        N = x.shape[0]
        C = x.shape[1]
        G = self.num_groups

        if x.ndim == 2:
            x_grouped = x.reshape(N, G, self.group_size)
            mean = np.mean(x_grouped, axis=(2,), keepdims=True)
            var = np.var(x_grouped, axis=(2,), keepdims=True)
            normalized = (x_grouped - mean) / np.sqrt(var + self.eps)
            normalized = normalized.reshape(N, C)
            out = normalized * self.gamma + self.beta
        elif x.ndim == 4:
            H, W = x.shape[2], x.shape[3]
            x_grouped = x.reshape(N, G, self.group_size, H, W)

            # Compute per-group mean/variance
            mean = np.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
            var = np.var(x_grouped, axis=(2, 3, 4), keepdims=True)

            # Normalize per group
            normalized = (x_grouped - mean) / np.sqrt(var + self.eps)
            normalized = normalized.reshape(N, C, H, W)

            # Apply learnable parameters
            out = normalized * self.gamma.reshape(1, C, 1, 1) + self.beta.reshape(1, C, 1, 1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        self.normalized = normalized
        self.mean = mean
        self.var = var
        return out

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        x = self.x
        G = self.num_groups
        group_size = self.group_size

        if x.ndim == 2:
            N, C = x.shape
            x_grouped = x.reshape(N, G, group_size)
            grad_norm = previous_grad * self.gamma
            grad_norm = grad_norm.reshape(N, G, group_size)
            mean = self.mean
            var = self.var
            std_inv = 1.0 / np.sqrt(var + self.eps)

            self.gamma_grad = np.sum(previous_grad * self.normalized, axis=0)
            self.beta_grad = np.sum(previous_grad, axis=0)

            grad_input = (1. / group_size) * std_inv * (
                group_size * grad_norm
                - np.sum(grad_norm, axis=2, keepdims=True)
                - (x_grouped - mean) * np.sum(grad_norm * (x_grouped - mean), axis=2, keepdims=True) * (std_inv ** 2)
            )
            grad_input = grad_input.reshape(N, C)
        elif x.ndim == 4:
            N, C, H, W = x.shape
            x_grouped = x.reshape(N, G, group_size, H, W)
            grad_norm = previous_grad * self.gamma.reshape(1, C, 1, 1)
            grad_norm = grad_norm.reshape(N, G, group_size, H, W)
            mean = self.mean
            var = self.var
            std_inv = 1.0 / np.sqrt(var + self.eps)

            self.gamma_grad = np.sum(previous_grad * self.normalized, axis=(0, 2, 3))
            self.beta_grad = np.sum(previous_grad, axis=(0, 2, 3))

            m = group_size * H * W
            grad_input = (1. / m) * std_inv * (
                m * grad_norm
                - np.sum(grad_norm, axis=(2, 3, 4), keepdims=True)
                - (x_grouped - mean)
                  * np.sum(grad_norm * (x_grouped - mean), axis=(2, 3, 4), keepdims=True)
                  * (std_inv ** 2)
            )
            grad_input = grad_input.reshape(N, C, H, W)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        return grad_input