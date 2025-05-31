from typing import Tuple
import numpy as np
from ...layers import Layer
from ...activations import Sigmoid

class GRUCell(Layer):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize GRUCell parameters

        Parameters:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize weights using Xavier uniform
        limit = np.sqrt(6 / (input_size + hidden_size))
        # Update gate
        self.W_z = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_z = np.zeros(hidden_size, dtype=np.float32)
        # Reset gate
        self.W_r = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_r = np.zeros(hidden_size, dtype=np.float32)
        # Candidate hidden
        self.W_h = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_h = np.zeros(hidden_size, dtype=np.float32)

        # Gradients
        self.dW_z = np.zeros_like(self.W_z, dtype=np.float32)
        self.db_z = np.zeros_like(self.b_z, dtype=np.float32)
        self.dW_r = np.zeros_like(self.W_r, dtype=np.float32)
        self.db_r = np.zeros_like(self.b_r, dtype=np.float32)
        self.dW_h = np.zeros_like(self.W_h, dtype=np.float32)
        self.db_h = np.zeros_like(self.b_h, dtype=np.float32)

        # Cache for backward
        self.cache = None

    def parameters(self):
        yield self.W_z
        yield self.b_z
        yield self.W_r
        yield self.b_r
        yield self.W_h
        yield self.b_h

    def gradients(self):
        yield self.dW_z
        yield self.db_z
        yield self.dW_r
        yield self.db_r
        yield self.dW_h
        yield self.db_h

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for single GRU cell

        Parameters:
            x: Input at current time-step
            h_prev: Previous hidden state

        Returns:
            h_next: Next hidden state
        """
        concat = np.concatenate([h_prev, x], axis=1)
        z = Sigmoid(concat.dot(self.W_z.T) + self.b_z)
        r = Sigmoid(concat.dot(self.W_r.T) + self.b_r)
        concat_reset = np.concatenate([r * h_prev, x], axis=1)
        h_hat = np.tanh(concat_reset.dot(self.W_h.T) + self.b_h)
        h_next = (1 - z) * h_prev + z * h_hat
        # Cache for backward
        self.cache = (x, h_prev, z, r, h_hat, concat, concat_reset)
        return h_next

    def backward(self, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for single GRU cell

        Parameters:
            dh_next: Gradient of loss w.r.t. next hidden state

        Returns:
            dx: Gradient w.r.t. input x
            dh_prev: Gradient w.r.t. previous hidden state
        """
        _, h_prev, z, r, h_hat, concat, concat_reset = self.cache
        # Gradients w.r.t. h_hat and z
        dh_hat = dh_next * z
        dz = dh_next * (h_hat - h_prev)
        # Backprop through h_hat
        dh_hat_input = dh_hat * (1 - h_hat ** 2)
        self.dW_h = dh_hat_input.T.dot(concat_reset)
        self.db_h = np.sum(dh_hat_input, axis=0)
        d_concat_reset = dh_hat_input.dot(self.W_h)
        # Split concat_reset gradient
        dr_h_prev = d_concat_reset[:, :self.hidden_size]
        dx_h = d_concat_reset[:, self.hidden_size:]
        # Backprop through reset gate
        dr = dr_h_prev * h_prev
        dr_input = dr * r * (1 - r)
        self.dW_r = dr_input.T.dot(concat)
        self.db_r = np.sum(dr_input, axis=0)
        d_concat_r = dr_input.dot(self.W_r)
        dh_prev_r = d_concat_r[:, :self.hidden_size]
        dx_r = d_concat_r[:, self.hidden_size:]
        # Backprop through update gate
        dz_input = dz * z * (1 - z)
        self.dW_z = dz_input.T.dot(concat)
        self.db_z = np.sum(dz_input, axis=0)
        d_concat_z = dz_input.dot(self.W_z)
        dh_prev_z = d_concat_z[:, :self.hidden_size]
        dx_z = d_concat_z[:, self.hidden_size:]
        # Aggregate gradients for h_prev
        dh_prev = dh_next * (1 - z) + dh_prev_z + dh_prev_r
        # Gradient w.r.t x
        dx = dx_h + dx_r + dx_z
        return dx, dh_prev

class GRU(Layer):
    """
    Multi-step GRU for sequence processing
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize GRU parameters

        Parameters:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        self.cache = None

    def parameters(self):
        yield from self.cell.parameters()

    def gradients(self):
        yield from self.cell.gradients()

    def forward(self, 
                x: np.ndarray, 
                h0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for entire sequence using GRU

        Parameters:
            x: Input sequence
            h0: Initial hidden state

        Returns:
            outputs: Hidden states for all time-steps
            h_last: Final hidden state
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            h_prev = h0
        outputs = []
        self.cache = []
        for t in range(seq_len):
            xt = x[:, t, :]
            h_next = self.cell.forward(xt, h_prev)
            outputs.append(h_next[:, np.newaxis, :])
            self.cache.append(h_prev)
            h_prev = h_next
        outputs = np.concatenate(outputs, axis=1)
        self.cache.append(x)
        return outputs, h_prev

    def backward(self, grad_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for entire sequence using GRU

        Parameters:
            grad_outputs: Gradient of loss w.r.t. outputs

        Returns:
            grad_input: Gradient w.r.t. input sequence
            dh0: Gradient w.r.t. initial hidden state
        """
        batch_size, seq_len, _ = grad_outputs.shape
        grad_input = np.zeros((batch_size, seq_len, self.input_size), dtype=np.float32)
        dh_next = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        x = self.cache[-1]
        for t in reversed(range(seq_len)):
            h_prev = self.cache[t]
            dh = grad_outputs[:, t, :] + dh_next
            # Cell backward
            self.cell.cache = (x[:, t, :], h_prev, *self.cell.cache[2:])  # Reconstruct cache
            dx_t, dh_prev = self.cell.backward(dh)
            grad_input[:, t, :] = dx_t
            dh_next = dh_prev
        dh0 = dh_next
        return grad_input, dh0
