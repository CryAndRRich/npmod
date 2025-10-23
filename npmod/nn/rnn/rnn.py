from typing import Tuple
import numpy as np
from ..layers import Layer

class RNNCell(Layer):
    """
    Single recurrent neural network cell
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize RNNCell parameters

        Parameters:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weights and biases
        self.W_x = np.zeros((hidden_size, input_size), dtype=np.float32)
        self.W_h = np.zeros((hidden_size, hidden_size), dtype=np.float32)
        self.b_ih = np.zeros(hidden_size, dtype=np.float32)
        self.b_hh = np.zeros(hidden_size, dtype=np.float32)
        
        self.dW_x = np.zeros_like(self.W_x)
        self.dW_h = np.zeros_like(self.W_h)
        self.db_ih = np.zeros_like(self.b_ih)
        self.db_hh = np.zeros_like(self.b_hh)

        # Cache for backward
        self.cache = None

    def parameters(self):
        yield self.W_x
        yield self.W_h
        yield self.b_ih
        yield self.b_hh

    def gradients(self):
        yield self.dW_x
        yield self.dW_h
        yield self.db_ih
        yield self.db_hh

    def forward(self, 
                x: np.ndarray, 
                h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for single time-step

        Parameters:
            x: Input at current time-step
            h_prev: Previous hidden state

        Returns:
            h_next: Next hidden state
        """
        a = x.dot(self.W_x.T) + self.b_ih + h_prev.dot(self.W_h.T) + self.b_hh
        h_next = np.tanh(a)

        # Cache for backward
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for single time-step

        Parameters:
            dh_next: Gradient of loss with respect to next hidden state

        Returns:
            dx: Gradient with respect to input x
            dh_prev: Gradient with respect to previous hidden state
        """
        x, h_prev, h_next = self.cache
        # Backprop through tanh
        da = dh_next * (1 - h_next ** 2)

        # Gradients
        self.dW_x += da.T.dot(x)
        self.dW_h += da.T.dot(h_prev)
        self.db_ih += np.sum(da, axis=0)
        self.db_hh += np.sum(da, axis=0)

        dx = da.dot(self.W_x)
        dh_prev = da.dot(self.W_h)
        return dx, dh_prev

class RNN(Layer):
    """
    Multilayer RNN for sequence processing
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize RNN parameters

        Parameters:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Single RNNCell
        self.cell = RNNCell(input_size, hidden_size)

        # Cache for backward
        self.cache = None

    def parameters(self):
        yield from self.cell.parameters()

    def gradients(self):
        yield from self.cell.gradients()

    def forward(self, 
                x: np.ndarray, 
                h0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for entire sequence

        Parameters:
            x: Input sequence
            h0: Initial hidden state. If None, initialized to zeros

        Returns:
            outputs: Hidden states for all time-steps
            h_last: Final hidden state
        """
        batch, seq_len, _ = x.shape
        if h0 is None:
            h_prev = np.zeros((batch, self.hidden_size), dtype=np.float32)
        else:
            h_prev = h0

        outputs = []
        self.cache = []
        for t in range(seq_len):
            h_next = self.cell.forward(x[:, t, :], h_prev)
            outputs.append(h_next[:, np.newaxis, :])
            self.cache.append((x[:, t, :], h_prev, h_next))
            h_prev = h_next
        return np.concatenate(outputs, axis=1), h_prev

    def backward(self, grad_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for entire sequence

        Parameters:
            grad_outputs: Gradient of loss with respect to outputs

        Returns:
            grad_input: Gradient with respect to input sequence
            dh0: Gradient with respect to initial hidden state
        """
        batch, seq_len, _ = grad_outputs.shape
        grad_input = np.zeros((batch, seq_len, self.cell.W_x.shape[1]), dtype=np.float32)
        dh_next = np.zeros((batch, self.hidden_size), dtype=np.float32)

        # Reset grads
        self.cell.dW_x.fill(0)
        self.cell.dW_h.fill(0)
        self.cell.db_ih.fill(0)
        self.cell.db_hh.fill(0)

        for t in reversed(range(seq_len)):
            dh = grad_outputs[:, t, :] + dh_next
            self.cell.cache = self.cache[t]
            dx, dh_prev = self.cell.backward(dh)
            grad_input[:, t, :] = dx
            dh_next = dh_prev

        return grad_input, dh_next