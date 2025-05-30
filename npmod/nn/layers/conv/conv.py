import numpy as np
from ..conv import _padding
from ...layers import Layer

class Conv2D(Layer):
    """
    2D Convolutional Layer
    """
    def __init__(self, 
                 kernel: np.ndarray, 
                 stride: int = 1, 
                 padding: int = 0) -> None:
        """
        Parameters:
            kernel: The 2D kernel matrix used for convolution
            stride: Stride of the convolution operation
            padding: Number of zero-padding layers to add to the input
        """
        super().__init__()
        assert len(kernel.shape) == 2 and kernel.shape[0] == kernel.shape[1], \
            "The kernel must be a square matrix"

        self.stride = stride
        self.padding = padding
        
        self.kernel = kernel
        self.kernel_size = kernel.shape[0]
        self.kernel_grad = np.zeros_like(kernel, dtype=np.float32)
        
        # Store the padded input for the backward pass
        self.input = None

    def parameters(self):
        yield self.kernel

    def gradients(self):
        yield self.kernel_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2 and x.shape[0] == x.shape[1], "Input must be a square matrix"
        assert x.shape[0] >= self.kernel_size, "The input must be larger than the kernel"
        
        # Apply padding to the input if needed
        x_padded = _padding(x, self.padding)

        padded_size = x_padded.shape[0]
        if padded_size < self.kernel_size:
            raise Exception("The padded input is smaller than the kernel")
        
        # Store the padded input for the backward pass
        self.input = x_padded.copy()
        
        # Flip the kernel
        flipped_kernel = np.flip(self.kernel)
        
        # Compute the dimensions of the output
        output_size = (padded_size - self.kernel_size) // self.stride + 1
        output = np.zeros((output_size, output_size), dtype=np.float32)
        
        for i in range(output_size):
            for j in range(output_size):
                row_start = i * self.stride
                row_end = row_start + self.kernel_size
                col_start = j * self.stride
                col_end = col_start + self.kernel_size
                patch = x_padded[row_start:row_end, col_start:col_end]
                conv_sum = np.sum(patch * flipped_kernel)
                output[i, j] = conv_sum
        
        return output

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        assert len(previous_grad.shape) == 2, "Gradient must be 2D"
        
        output_size = previous_grad.shape[0]
        
        # Initialize gradient with respect to the padded input
        input_grad_padded = np.zeros_like(self.input, dtype=np.float32)
        kernel_grad_flipped = np.zeros_like(self.kernel, dtype=np.float32)
        
        # Flip the kernel for the backward computation
        flipped_kernel = np.flip(self.kernel)
        
        for i in range(output_size):
            for j in range(output_size):
                row_start = i * self.stride
                row_end = row_start + self.kernel_size
                col_start = j * self.stride
                col_end = col_start + self.kernel_size
                # Accumulate gradient with respect to the input
                input_grad_padded[row_start:row_end, col_start:col_end] += previous_grad[i, j] * flipped_kernel
                # Accumulate gradient with respect to the kernel
                patch = self.input[row_start:row_end, col_start:col_end]
                kernel_grad_flipped += previous_grad[i, j] * patch
        
        # Compute the gradient with respect to the kernel
        self.kernel_grad = np.flip(kernel_grad_flipped)
        
        # Remove padding from the input gradient if necessary
        if self.padding > 0:
            input_grad = input_grad_padded[self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_grad = input_grad_padded
        
        return input_grad

class Conv3D(Layer):
    """
    3D Convolutional Layer
    """
    def __init__(self, 
                 kernel: np.ndarray, 
                 stride: int = 1, 
                 padding: int = 0) -> None:
        """
        Parameters:
            kernel: The 3D kernel used for convolution
            stride: Stride of the convolution operation
            padding: Number of zero-padding layers to add to the input
        """
        super().__init__()
        # Ensure the kernel is a 3D cube matrix
        assert len(kernel.shape) == 3 and kernel.shape[0] == kernel.shape[1] == kernel.shape[2], \
            "The kernel must be a cube matrix"
        
        self.stride = stride
        self.padding = padding

        self.kernel = kernel
        self.kernel_size = kernel.shape[0]
        self.kernel_grad = np.zeros_like(kernel, dtype=np.float32)
        
        self.input = None

    def parameters(self):
        yield self.kernel

    def gradients(self):
        yield self.kernel_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Ensure the input is non-empty and a cube matrix
        assert len(x.shape) == 3 and x.shape[0] == x.shape[1] == x.shape[2], \
            "Input must be a cube matrix"

        x_padded = _padding(x, self.padding)

        padded_dim = x_padded.shape[0]
        if padded_dim < self.kernel_size:
            raise Exception("The padded input is smaller than the kernel.")

        self.input = x_padded.copy()

        flipped_kernel = np.flip(self.kernel)

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
                    conv_sum = np.sum(patch * flipped_kernel)
                    output[d, i, j] = conv_sum

        return output

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        assert len(previous_grad.shape) == 3, "The gradient matrix must be 3D"

        output_dim = previous_grad.shape[0]

        input_grad_padded = np.zeros_like(self.input, dtype=np.float32)
        kernel_grad_flipped = np.zeros_like(self.kernel, dtype=np.float32)

        flipped_kernel = np.flip(self.kernel)

        for d in range(output_dim):
            for i in range(output_dim):
                for j in range(output_dim):
                    d_start = d * self.stride
                    d_end = d_start + self.kernel_size
                    i_start = i * self.stride
                    i_end = i_start + self.kernel_size
                    j_start = j * self.stride
                    j_end = j_start + self.kernel_size

                    input_grad_padded[d_start:d_end, i_start:i_end, j_start:j_end] += previous_grad[d, i, j] * flipped_kernel

                    patch = self.input[d_start:d_end, i_start:i_end, j_start:j_end]
                    kernel_grad_flipped += previous_grad[d, i, j] * patch

        self.kernel_grad = np.flip(kernel_grad_flipped)

        if self.padding > 0:
            input_grad = input_grad_padded[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_grad = input_grad_padded

        return input_grad
