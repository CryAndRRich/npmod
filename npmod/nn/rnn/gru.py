from typing import Tuple
import numpy as np
from ..layers import Layer

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
        self.input_size = input_size
        self.hidden_size = hidden_size

        limit = np.sqrt(6 / (input_size + hidden_size))

        # Update gate
        self.W_z = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_z = np.zeros(hidden_size, dtype=np.float32)

        # Reset gate
        self.W_r = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_r = np.zeros(hidden_size, dtype=np.float32)

        # Candidate hidden (reset_after=True)
        self.W_h = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_h_ih = np.zeros(hidden_size, dtype=np.float32)
        self.b_h_hh = np.zeros(hidden_size, dtype=np.float32)

        # Gradients
        self.dW_z = np.zeros_like(self.W_z)
        self.db_z = np.zeros_like(self.b_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.db_r = np.zeros_like(self.b_r)
        self.dW_h = np.zeros_like(self.W_h)
        self.db_h_ih = np.zeros_like(self.b_h_ih)
        self.db_h_hh = np.zeros_like(self.b_h_hh)

        self.cache = None

    def parameters(self):
        yield self.W_z
        yield self.b_z
        yield self.W_r
        yield self.b_r
        yield self.W_h
        yield self.b_h_ih
        yield self.b_h_hh

    def gradients(self):
        yield self.dW_z
        yield self.db_z
        yield self.dW_r
        yield self.db_r
        yield self.dW_h
        yield self.db_h_ih
        yield self.db_h_hh

    def forward(self, 
                x: np.ndarray, 
                h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for single GRU cell

        Parameters:
            x: Input at current time-step
            h_prev: Previous hidden state

        Returns:
            h_next: Next hidden state
        """
        # Split weights
        Wz_x, Wz_h = self.W_z[:, self.hidden_size:], self.W_z[:, :self.hidden_size]
        Wr_x, Wr_h = self.W_r[:, self.hidden_size:], self.W_r[:, :self.hidden_size]
        Wh_x, Wh_h = self.W_h[:, self.hidden_size:], self.W_h[:, :self.hidden_size]

        z = self._sigmoid(x @ Wz_x.T + h_prev @ Wz_h.T + self.b_z)
        r = self._sigmoid(x @ Wr_x.T + h_prev @ Wr_h.T + self.b_r)

        h_hat = np.tanh(x @ Wh_x.T + self.b_h_ih + r * (h_prev @ Wh_h.T + self.b_h_hh))

        h_next = (1 - z) * h_hat + z * h_prev

        self.cache = (x, h_prev, z, r, h_hat, Wz_x, Wz_h, Wr_x, Wr_h, Wh_x, Wh_h)
        return h_next

    def backward(self, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for single GRU cell

        Parameters:
            dh_next: Gradient of loss with respect to next hidden state

        Returns:
            dx: Gradient with respect to input x
            dh_prev: Gradient with respect to previous hidden state
        """
        x, h_prev, z, r, h_hat, Wz_x, Wz_h, Wr_x, Wr_h, Wh_x, Wh_h = self.cache

        dh_hat = dh_next * (1 - z)         
        dz = dh_next * (h_prev - h_hat)      
        dh_prev_through_z = dh_next * z        

        # Backprop through tanh
        dh_hat_pre = dh_hat * (1 - h_hat ** 2)   

        u = h_prev @ Wh_h.T + self.b_h_hh

        self.dW_h[:, self.hidden_size:] += dh_hat_pre.T @ x              
        self.dW_h[:, :self.hidden_size] += (dh_hat_pre * r).T @ h_prev   

        self.db_h_ih += np.sum(dh_hat_pre, axis=0)
        self.db_h_hh += np.sum(dh_hat_pre * r, axis=0)

        dx_h = dh_hat_pre @ Wh_x

        dh_prev_h_part = (dh_hat_pre * r) @ Wh_h

        # Backprop through r
        dr = dh_hat_pre * u
        dr_input = dr * r * (1 - r)

        self.dW_r[:, self.hidden_size:] += dr_input.T @ x
        self.dW_r[:, :self.hidden_size] += dr_input.T @ h_prev
        self.db_r += np.sum(dr_input, axis=0)

        dx_r = dr_input @ Wr_x
        dh_prev_r = dr_input @ Wr_h

        # Backprop through z
        dz_input = dz * z * (1 - z)
        self.dW_z[:, self.hidden_size:] += dz_input.T @ x
        self.dW_z[:, :self.hidden_size] += dz_input.T @ h_prev
        self.db_z += np.sum(dz_input, axis=0)

        dx_z = dz_input @ Wz_x
        dh_prev_z = dz_input @ Wz_h

        dx = dx_h + dx_r + dx_z
        dh_prev = dh_prev_through_z + dh_prev_h_part + dh_prev_r + dh_prev_z

        return dx, dh_prev
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

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
            # Store full cell cache for this timestep
            self.cache.append(self.cell.cache)
            h_prev = h_next
        outputs = np.concatenate(outputs, axis=1)
        return outputs, h_prev

    def backward(self, grad_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for entire sequence using GRU

        Parameters:
            grad_outputs: Gradient of loss with respect to outputs

        Returns:
            grad_input: Gradient with respect to input sequence
            dh0: Gradient with respect to initial hidden state
        """
        self.cell.dW_z.fill(0)
        self.cell.db_z.fill(0)
        self.cell.dW_r.fill(0)
        self.cell.db_r.fill(0)
        self.cell.dW_h.fill(0)
        self.cell.db_h_ih.fill(0)
        self.cell.db_h_hh.fill(0)

        batch_size, seq_len, _ = grad_outputs.shape
        grad_input = np.zeros((batch_size, seq_len, self.input_size), dtype=np.float32)
        dh_next = np.zeros((batch_size, self.hidden_size), dtype=np.float32)

        for t in reversed(range(seq_len)):
            dh = grad_outputs[:, t, :] + dh_next
            # Set cell cache for this timestep
            self.cell.cache = self.cache[t]
            dx_t, dh_prev = self.cell.backward(dh)
            grad_input[:, t, :] = dx_t
            dh_next = dh_prev

        dh0 = dh_next
        return grad_input, dh0
