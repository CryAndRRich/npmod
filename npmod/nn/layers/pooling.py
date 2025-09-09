import numpy as np
from ..layers import Layer

class Pooling(Layer):
    """
    General Pooling Layer (supports Max/Avg, 2D/3D, and Global pooling)
    """
    def __init__(self,
                 kernel_size: int = None,
                 stride: int = None,
                 padding: int = 0,
                 num_dims: int = 2,
                 mode: str = "max",
                 global_pool: bool = False) -> None:
        """
        Parameters:
            kernel_size: Size of pooling window (ignored if global_pool=True)
            stride: Stride of pooling (default = kernel_size, ignored if global_pool=True)
            padding: Zero padding around input
            num_dims: 2 for 2D, 3 for 3D
            mode: "max" or "avg"
            global_pool: If True, performs global pooling over full input
        """
        super().__init__()
        assert num_dims in (2, 3), "num_dims must be 2 or 3"
        assert mode in ("max", "avg"), "mode must be 'max' or 'avg'"
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.num_dims = num_dims
        self.mode = mode
        self.global_pool = global_pool

        # buffers saved for backward
        self.input = None
        self._input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Expect batched input
        if self.num_dims == 2:
            assert x.ndim == 4, "For 2D pooling input must be (N, C, H, W)"
            N, C, H, W = x.shape
            k = H if self.global_pool else self.kernel_size
            s = 1 if self.global_pool else self.stride
            p = self.padding
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
            self.input = x_padded.copy()
            self._input_shape = x.shape
            H_out = (H + 2 * p - k) // s + 1
            W_out = (W + 2 * p - k) // s + 1
            out = np.zeros((N, C, H_out, W_out), dtype=np.float32)

            # naive loop
            for n in range(N):
                for c in range(C):
                    for i in range(H_out):
                        for j in range(W_out):
                            r0, r1 = i * s, i * s + k
                            c0, c1 = j * s, j * s + k
                            patch = x_padded[n, c, r0:r1, c0:c1]
                            if self.mode == "max":
                                out[n, c, i, j] = np.max(patch)
                            else:
                                out[n, c, i, j] = np.mean(patch)
            return out

        else: 
            assert x.ndim == 5, "For 3D pooling input must be (N, C, D, H, W)"
            N, C, D, H, W = x.shape
            k = D if self.global_pool else self.kernel_size
            s = 1 if self.global_pool else self.stride
            p = self.padding
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p), (p, p)), mode='constant')
            self.input = x_padded.copy()
            self._input_shape = x.shape
            D_out = (D + 2 * p - k) // s + 1
            H_out = (H + 2 * p - k) // s + 1
            W_out = (W + 2 * p - k) // s + 1
            out = np.zeros((N, C, D_out, H_out, W_out), dtype=np.float32)

            for n in range(N):
                for c in range(C):
                    for d in range(D_out):
                        for i in range(H_out):
                            for j in range(W_out):
                                d0, d1 = d * s, d * s + k
                                r0, r1 = i * s, i * s + k
                                c0, c1 = j * s, j * s + k
                                patch = x_padded[n, c, d0:d1, r0:r1, c0:c1]
                                if self.mode == "max":
                                    out[n, c, d, i, j] = np.max(patch)
                                else:
                                    out[n, c, d, i, j] = np.mean(patch)
            return out

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        # previous_grad shape matches forward output shape
        p = self.padding
        k = self._input_shape[2] if self.global_pool and self.num_dims==2 else self.kernel_size

        if self.num_dims == 2:
            N, C, H, _ = self._input_shape
            k = H if self.global_pool else self.kernel_size
            s = 1 if self.global_pool else self.stride
            p = self.padding
            _, _, H_out, W_out = previous_grad.shape
            input_grad_padded = np.zeros_like(self.input, dtype=np.float32)

            for n in range(N):
                for c in range(C):
                    for i in range(H_out):
                        for j in range(W_out):
                            r0, r1 = i * s, i * s + k
                            c0, c1 = j * s, j * s + k
                            patch = self.input[n, c, r0:r1, c0:c1]
                            if self.mode == "max":
                                max_val = np.max(patch)
                                mask = (patch == max_val).astype(np.float32)
                                input_grad_padded[n, c, r0:r1, c0:c1] += previous_grad[n, c, i, j] * mask
                            else:
                                grad_share = previous_grad[n, c, i, j] / (k * k)
                                input_grad_padded[n, c, r0:r1, c0:c1] += grad_share
            if p > 0:
                return input_grad_padded[:, :, p:-p, p:-p]
            return input_grad_padded

        else: 
            N, C, D, H, _ = self._input_shape
            k = D if self.global_pool else self.kernel_size
            s = 1 if self.global_pool else self.stride
            p = self.padding
            _, _, D_out, H_out, W_out = previous_grad.shape
            input_grad_padded = np.zeros_like(self.input, dtype=np.float32)

            for n in range(N):
                for c in range(C):
                    for d in range(D_out):
                        for i in range(H_out):
                            for j in range(W_out):
                                d0, d1 = d * s, d * s + k
                                r0, r1 = i * s, i * s + k
                                c0, c1 = j * s, j * s + k
                                patch = self.input[n, c, d0:d1, r0:r1, c0:c1]
                                if self.mode == "max":
                                    max_val = np.max(patch)
                                    mask = (patch == max_val).astype(np.float32)
                                    input_grad_padded[n, c, d0:d1, r0:r1, c0:c1] += previous_grad[n, c, d, i, j] * mask
                                else:
                                    grad_share = previous_grad[n, c, d, i, j] / (k ** 3)
                                    input_grad_padded[n, c, d0:d1, r0:r1, c0:c1] += grad_share
            if p > 0:
                return input_grad_padded[:, :, p:-p, p:-p, p:-p]
            return input_grad_padded