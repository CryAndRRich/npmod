import numpy as np
from ..conv import _padding
from ...layers import Layer

class MaxPool2D(Layer):
    """
    2D Max-Pooling Layer
    """
    def __init__(self, 
                 kernel_size: int, 
                 stride: int = None, 
                 padding: int = 0) -> None:
        """
        Parameters:
            kernel_size: The size of the pooling window
            stride: Stride of the pooling operation
            padding: Number of zero-padding layers to add to the input
        """
        super().__init__()

        # If stride is not provided, use kernel_size (non-overlapping pooling windows)
        if stride is None:
            stride = kernel_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Store the padded input for the backward pass
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Ensure the input is non-empty and a square matrix
        assert len(x.shape) == 2 and x.shape[0] == x.shape[1], \
            "Input must be a square matrix"
        assert x.shape[0] >= self.kernel_size, \
            "Input must be larger than the pooling window"

        # Apply padding to the input if needed
        x_padded = _padding(x, self.padding)

        padded_size = x_padded.shape[0]
        if padded_size < self.kernel_size:
            raise Exception("The padded input is smaller than the pooling window")

        # Store the padded input for the backward pass.
        self.input = x_padded.copy()

        # Compute the dimensions of the output.
        output_size = (padded_size - self.kernel_size) // self.stride + 1
        output = np.zeros((output_size, output_size), dtype=np.float32)

        for i in range(output_size):
            for j in range(output_size):
                row_start = i * self.stride
                row_end = row_start + self.kernel_size
                col_start = j * self.stride
                col_end = col_start + self.kernel_size
                patch = x_padded[row_start:row_end, col_start:col_end]
                output[i, j] = np.max(patch)

        return output

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        assert len(previous_grad.shape) == 2, "The gradient matrix must be 2D"

        output_size = previous_grad.shape[0]

        input_grad_padded = np.zeros_like(self.input, dtype=np.float32)

        for i in range(output_size):
            for j in range(output_size):
                row_start = i * self.stride
                row_end = row_start + self.kernel_size
                col_start = j * self.stride
                col_end = col_start + self.kernel_size

                # Extract the patch corresponding to the pooling window
                patch = self.input[row_start:row_end, col_start:col_end]
                # Create a mask for the maximum value in the patch
                mask = (patch == np.max(patch)).astype(np.float32)
                # Route the gradient to the position(s) of the maximum value
                input_grad_padded[row_start:row_end, col_start:col_end] += previous_grad[i, j] * mask

        # Remove padding from the gradient if necessary
        if self.padding > 0:
            input_grad = input_grad_padded[self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_grad = input_grad_padded

        return input_grad

class MaxPool3D(Layer):
    """
    3D Max-Pooling Layer
    """
    def __init__(self, 
                 kernel_size: int, 
                 stride: int = None, 
                 padding: int = 0) -> None:
        """
        Parameters:
            kernel_size: The size of the pooling window
            stride: Stride of the pooling operation
            padding: Number of zero-padding layers to add to the input
        """
        super().__init__()

        if stride is None:
            stride = kernel_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 3 and x.shape[0] == x.shape[1] == x.shape[2], \
            "Input is not a cube matrix"
        assert x.shape[0] >= self.kernel_size, \
            "Input must be larger than the pooling window"

        x_padded = _padding(x, self.padding)

        padded_dim = x_padded.shape[0]
        if padded_dim < self.kernel_size:
            raise Exception("The padded input is smaller than the pooling window")

        self.input = x_padded.copy()

        output_dim = (padded_dim - self.kernel_size) // self.stride + 1
        output = np.zeros((output_dim, output_dim, output_dim), dtype=np.float32)

        for d in range(output_dim):
            for i in range(output_dim):
                for j in range(output_dim):
                    d_start = d * self.stride
                    d_end = d_start + self.kernel_size
                    i_start = i * self.stride
                    i_end = i_start + self.kernel_size
                    j_start = j * self.stride
                    j_end = j_start + self.kernel_size

                    patch = x_padded[d_start:d_end, i_start:i_end, j_start:j_end]
                    output[d, i, j] = np.max(patch)

        return output

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        assert len(previous_grad.shape) == 3, "The gradient matrix must be 3D"

        output_dim = previous_grad.shape[0]

        input_grad_padded = np.zeros_like(self.input, dtype=np.float32)

        for d in range(output_dim):
            for i in range(output_dim):
                for j in range(output_dim):
                    d_start = d * self.stride
                    d_end = d_start + self.kernel_size
                    i_start = i * self.stride
                    i_end = i_start + self.kernel_size
                    j_start = j * self.stride
                    j_end = j_start + self.kernel_size

                    patch = self.input[d_start:d_end, i_start:i_end, j_start:j_end]
                    mask = (patch == np.max(patch)).astype(np.float32)
                    input_grad_padded[d_start:d_end, i_start:i_end, j_start:j_end] += previous_grad[d, i, j] * mask

        if self.padding > 0:
            input_grad = input_grad_padded[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_grad = input_grad_padded

        return input_grad
