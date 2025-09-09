import numpy as np
from ..layers import Layer

class Conv(Layer):
    """
    Generalized Convolution Layer supporting both 2D and 3D
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 num_dims: int = 2) -> None:
        """
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel (assumed square/cubic)
            stride: Stride of the convolution
            padding: Padding added to all sides of the input
            num_dims: 2 for Conv2D, 3 for Conv3D
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_dims = num_dims

        # Initialize kernel and bias
        if num_dims == 2:
            self.weight = (np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01).astype(np.float32)
        else:
            self.weight = (np.random.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size) * 0.01).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)

        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

        self.input = None
        self.input_pad = None

    def parameters(self):
        yield self.weight
        yield self.bias

    def gradients(self):
        yield self.weight_grad
        yield self.bias_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        N = x.shape[0]

        if self.num_dims == 2:
            _, C, H, W = x.shape
            assert C == self.in_channels
            H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
            W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
            out = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)

            # Pad each sample
            x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            self.input_pad = x_pad.copy()

            for n in range(N):
                for oc in range(self.out_channels):
                    for ic in range(self.in_channels):
                        for i in range(H_out):
                            for j in range(W_out):
                                i0, i1 = i * self.stride, i * self.stride + self.kernel_size
                                j0, j1 = j * self.stride, j * self.stride + self.kernel_size
                                out[n, oc, i, j] += np.sum(x_pad[n, ic, i0:i1, j0:j1] * self.weight[oc, ic])
                    out[n, oc] += self.bias[oc]

        else: 
            _, C, D, H, W = x.shape
            assert C == self.in_channels
            D_out = (D + 2 * self.padding - self.kernel_size) // self.stride + 1
            H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
            W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
            out = np.zeros((N, self.out_channels, D_out, H_out, W_out), dtype=np.float32)

            x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            self.input_pad = x_pad.copy()

            for n in range(N):
                for oc in range(self.out_channels):
                    for ic in range(self.in_channels):
                        for d in range(D_out):
                            for i in range(H_out):
                                for j in range(W_out):
                                    d0, d1 = d * self.stride, d * self.stride + self.kernel_size
                                    i0, i1 = i * self.stride, i * self.stride + self.kernel_size
                                    j0, j1 = j * self.stride, j * self.stride + self.kernel_size
                                    out[n, oc, d, i, j] += np.sum(x_pad[n, ic, d0:d1, i0:i1, j0:j1] * self.weight[oc, ic])
                        out[n, oc] += self.bias[oc]

        return out

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        N = previous_grad.shape[0]

        if self.num_dims == 2:
            _, _, H_out, W_out = previous_grad.shape
            input_grad_padded = np.zeros_like(self.input_pad)
            self.weight_grad.fill(0)
            self.bias_grad.fill(0)

            for n in range(N):
                for oc in range(self.out_channels):
                    # bias gradient
                    self.bias_grad[oc] += np.sum(previous_grad[n, oc])
                    for ic in range(self.in_channels):
                        for i in range(H_out):
                            for j in range(W_out):
                                i0, i1 = i * self.stride, i * self.stride + self.kernel_size
                                j0, j1 = j * self.stride, j * self.stride + self.kernel_size
                                patch = self.input_pad[n, ic, i0:i1, j0:j1]
                                # kernel gradient
                                self.weight_grad[oc, ic] += previous_grad[n, oc, i, j] * patch
                                # input gradient
                                input_grad_padded[n, ic, i0:i1, j0:j1] += previous_grad[n, oc, i, j] * self.weight[oc, ic]

            # Remove padding
            if self.padding > 0:
                input_grad = input_grad_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                input_grad = input_grad_padded

        else:  
            _, _, D_out, H_out, W_out = previous_grad.shape
            input_grad_padded = np.zeros_like(self.input_pad)
            self.weight_grad.fill(0)
            self.bias_grad.fill(0)

            for n in range(N):
                for oc in range(self.out_channels):
                    # bias gradient
                    self.bias_grad[oc] += np.sum(previous_grad[n, oc])
                    for ic in range(self.in_channels):
                        for d in range(D_out):
                            for i in range(H_out):
                                for j in range(W_out):
                                    d0, d1 = d * self.stride, d * self.stride + self.kernel_size
                                    i0, i1 = i * self.stride, i * self.stride + self.kernel_size
                                    j0, j1 = j * self.stride, j * self.stride + self.kernel_size
                                    patch = self.input_pad[n, ic, d0:d1, i0:i1, j0:j1]
                                    # kernel gradient
                                    self.weight_grad[oc, ic] += previous_grad[n, oc, d, i, j] * patch
                                    # input gradient
                                    input_grad_padded[n, ic, d0:d1, i0:i1, j0:j1] += previous_grad[n, oc, d, i, j] * self.weight[oc, ic]

            # Remove padding
            if self.padding > 0:
                input_grad = input_grad_padded[:, :, self.padding:-self.padding,
                                               self.padding:-self.padding,
                                               self.padding:-self.padding]
            else:
                input_grad = input_grad_padded

        return input_grad
