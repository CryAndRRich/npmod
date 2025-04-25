import numpy as np
from ..layers import Layer

class BatchNorm(Layer):
    """
    Batch Normalization layer
    """
    def __init__(self, num_channels: int, 
                 momentum: float = 0.9, 
                 eps: float = 1e-6) -> None:
        """
        Parameters:
            num_channels: Number of channels (features) in the input
            momentum: Momentum factor for running statistics
            eps: Small constant to avoid division by zero
        """
        super().__init__()
        self.momentum = momentum
        self.eps = np.array([eps], dtype=np.float32)

        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = np.ones((num_channels,), dtype=np.float32)
        self.beta = np.zeros((num_channels,), dtype=np.float32)

        # Gradients of the learnable parameters
        self.gamma_grad = np.zeros((num_channels,), dtype=np.float32)
        self.beta_grad = np.zeros((num_channels,), dtype=np.float32)

        # Cached values for the forward pass
        self.input = np.zeros([], dtype=np.float32)
        self.batch_mean = np.array([], dtype=np.float32)
        self.batch_variance = np.array([], dtype=np.float32)
        self.normalized_input = np.array([], dtype=np.float32)

        # Running statistics for inference
        self.running_mean = np.array([0], dtype=np.float32)
        self.running_variance = np.array([0], dtype=np.float32)

    def parameters(self):
        yield self.gamma
        yield self.beta

    def gradients(self):
        yield self.gamma_grad
        yield self.beta_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2 or len(x.shape) == 3, "Input must be 2D or 3D"
        
        # Cache the input for use in the backward pass
        self.input = x.copy()
        # Get the size of the last dimension (used when handling 3D inputs)
        last_dim_size = self.input.shape[-1]

        if self.training:
            mean_var_axes = (2, 0) if len(self.input.shape) == 3 else (0,)

            # Compute batch mean and variance, keeping dimensions for broadcasting
            self.batch_mean = np.mean(self.input, axis=mean_var_axes, keepdims=True)
            self.batch_variance = np.var(self.input, axis=mean_var_axes, keepdims=True)

            # Normalize the input using the batch mean and variance
            self.normalized_input = (self.input - self.batch_mean) / np.sqrt(self.batch_variance + self.eps)

            # Apply the learnable scale (gamma) and shift (beta)
            if len(self.input.shape) == 3:
                output = (self.normalized_input * self.gamma[:, np.newaxis].repeat(last_dim_size, axis=-1)
                                 + self.beta[:, np.newaxis].repeat(last_dim_size, axis=-1))
            else:
                output = self.normalized_input * self.gamma + self.beta

            # Update the running statistics for inference
            self.running_mean = self.batch_mean * (1 - self.momentum) + self.running_mean * self.momentum
            self.running_variance = self.batch_variance * (1 - self.momentum) + self.running_variance * self.momentum

        else:
            # In inference mode, use the running mean and variance
            normalized_input = (self.input - self.running_mean) / np.sqrt(self.running_variance + self.eps)

            # Apply the scale and shift
            if len(self.input.shape) == 3:
                output = (normalized_input * self.gamma[:, np.newaxis].repeat(last_dim_size, axis=-1)
                                 + self.beta[:, np.newaxis].repeat(last_dim_size, axis=-1))
            else:
                output = normalized_input * self.gamma + self.beta

        return output

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        # Get the batch size (assumed to be the first dimension of the input)
        batch_size = self.input.shape[0]
        last_dim_size = self.input.shape[-1]

        # Compute the gradient with respect to the normalized input
        if len(self.input.shape) == 3:
            grad_normalized_input = previous_grad * self.gamma[:, np.newaxis].repeat(last_dim_size, axis=-1)
        else:
            grad_normalized_input = previous_grad * self.gamma

        grad_variance = np.sum((self.input - self.batch_mean) * grad_normalized_input * 
                               (-0.5) * (self.batch_variance + self.eps) ** (-3 / 2), axis=0)

        # Compute the gradient with respect to the mean
        grad_mean = (
            np.sum(grad_normalized_input * -1 / np.sqrt(self.batch_variance + self.eps), axis=0)
            + grad_variance * np.sum(-2 * (self.input - self.batch_mean), axis=0) / batch_size
        )

        # Combine the gradients to compute the gradient with respect to the input tensor
        grad_input = (grad_normalized_input / np.sqrt(self.batch_variance + self.eps)
                      + grad_variance * 2 * (self.input - self.batch_mean) / batch_size
                      + grad_mean / batch_size)

        # Determine the axes to sum over when computing parameter gradients
        sum_axes = (0, 2) if len(previous_grad.shape) == 3 else 0

        # Compute gradients for the learnable parameters
        self.gamma_grad = np.sum(previous_grad * self.normalized_input, axis=sum_axes).transpose()
        self.beta_grad = np.sum(previous_grad, axis=sum_axes).transpose()

        return grad_input
