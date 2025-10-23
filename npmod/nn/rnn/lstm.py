from typing import Tuple
import numpy as np
from ..layers import Layer

class LSTMCell(Layer):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize LSTMCell parameters

        Parameters:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize weights using Xavier uniform
        limit = np.sqrt(6 / (input_size + hidden_size))

        # Forget gate
        self.W_f = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_f = np.zeros(hidden_size, dtype=np.float32)
        # Input gate
        self.W_i = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_i = np.zeros(hidden_size, dtype=np.float32)
        # Cell gate
        self.W_c = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_c = np.zeros(hidden_size, dtype=np.float32)
        # Output gate
        self.W_o = np.random.uniform(-limit, limit, (hidden_size, input_size + hidden_size)).astype(np.float32)
        self.b_o = np.zeros(hidden_size, dtype=np.float32)

        # Gradients
        self.dW_f = np.zeros_like(self.W_f, dtype=np.float32)
        self.db_f = np.zeros_like(self.b_f, dtype=np.float32)
        self.dW_i = np.zeros_like(self.W_i, dtype=np.float32)
        self.db_i = np.zeros_like(self.b_i, dtype=np.float32)
        self.dW_c = np.zeros_like(self.W_c, dtype=np.float32)
        self.db_c = np.zeros_like(self.b_c, dtype=np.float32)
        self.dW_o = np.zeros_like(self.W_o, dtype=np.float32)
        self.db_o = np.zeros_like(self.b_o, dtype=np.float32)

        # Cache for backward
        self.cache = None

    def parameters(self):
        yield self.W_f
        yield self.b_f
        yield self.W_i
        yield self.b_i
        yield self.W_c
        yield self.b_c
        yield self.W_o
        yield self.b_o

    def gradients(self):
        yield self.dW_f
        yield self.db_f
        yield self.dW_i
        yield self.db_i
        yield self.dW_c
        yield self.db_c
        yield self.dW_o
        yield self.db_o

    def forward(self, 
                x: np.ndarray, 
                h_prev: np.ndarray, 
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for single LSTM cell

        Parameters:
            x: Input at current time-step
            h_prev: Previous hidden state
            c_prev: Previous cell state

        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        # Concatenate h_prev and x
        concat = np.concatenate([h_prev, x], axis=1)  # shape (batch, hidden_size + input_size)

        # Forget gate
        f = self._sigmoid(concat.dot(self.W_f.T) + self.b_f)
        # Input gate
        i = self._sigmoid(concat.dot(self.W_i.T) + self.b_i)
        # Cell gate
        c_hat = np.tanh(concat.dot(self.W_c.T) + self.b_c)
        # Output gate
        o = self._sigmoid(concat.dot(self.W_o.T) + self.b_o)

        c_next = f * c_prev + i * c_hat
        h_next = o * np.tanh(c_next)

        # Cache for backward
        self.cache = (x, h_prev, c_prev, f, i, c_hat, o, c_next, concat)
        return h_next, c_next

    def backward(self, 
                 dh_next: np.ndarray, 
                 dc_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for single LSTM cell

        Parameters:
            dh_next: Gradient of loss w.r.t. next hidden state
            dc_next: Gradient of loss w.r.t. next cell state

        Returns:
            dx: Gradient w.r.t. input x
            dh_prev: Gradient w.r.t. previous hidden state
            dc_prev: Gradient w.r.t. previous cell state
        """
        _, _, c_prev, f, i, c_hat, o, c_next, concat = self.cache

        tanh_c_next = np.tanh(c_next)
        do = dh_next * tanh_c_next
        do_input = do * o * (1 - o)

        dc = dc_next + dh_next * o * (1 - tanh_c_next ** 2)

        di = dc * c_hat
        di_input = di * i * (1 - i)

        dc_hat = dc * i
        dc_hat_input = dc_hat * (1 - c_hat ** 2)

        df = dc * c_prev
        df_input = df * f * (1 - f)

        d_concat = (do_input.dot(self.W_o)) + (di_input.dot(self.W_i)) + \
                (df_input.dot(self.W_f)) + (dc_hat_input.dot(self.W_c))

        dh_prev = d_concat[:, :self.hidden_size]
        dx = d_concat[:, self.hidden_size:]

        # Accumulate gradients
        self.dW_o += do_input.T.dot(concat)
        self.db_o += np.sum(do_input, axis=0)
        self.dW_i += di_input.T.dot(concat)
        self.db_i += np.sum(di_input, axis=0)
        self.dW_f += df_input.T.dot(concat)
        self.db_f += np.sum(df_input, axis=0)
        self.dW_c += dc_hat_input.T.dot(concat)
        self.db_c += np.sum(dc_hat_input, axis=0)

        dc_prev = dc * f
        return dx, dh_prev, dc_prev
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

class LSTM(Layer):
    """
    Multi-step LSTM for sequence processing
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize LSTM parameters

        Parameters:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
        self.cache = None

    def parameters(self):
        yield from self.cell.parameters()

    def gradients(self):
        yield from self.cell.gradients()

    def forward(self, 
                x: np.ndarray, 
                h0: np.ndarray = None, 
                c0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for entire sequence using LSTM

        Parameters:
            x: Input sequence
            h0: Initial hidden state
            c0: Initial cell state

        Returns:
            outputs: Hidden states for all time-steps
            (h_last, c_last): Final hidden and cell states
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            h_prev = h0
        if c0 is None:
            c_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            c_prev = c0

        outputs = []
        self.cache = []
        for t in range(seq_len):
            xt = x[:, t, :]
            h_next, c_next = self.cell.forward(xt, h_prev, c_prev)
            outputs.append(h_next[:, np.newaxis, :])
            self.cache.append(self.cell.cache)
            h_prev, c_prev = h_next, c_next
        outputs = np.concatenate(outputs, axis=1)
        self.cache.append(x)  # Store input sequence
        return outputs, h_next, c_next

    def backward(self, 
                 grad_outputs: np.ndarray, 
                 grad_h_last: np.ndarray = None, 
                 grad_c_last: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for entire sequence using LSTM

        Parameters:
            grad_outputs: Gradient of loss w.r.t. outputs
            grad_h_last: Gradient w.r.t. last hidden state
            grad_c_last: Gradient w.r.t. last cell state

        Returns:
            grad_input: Gradient w.r.t. input sequence
            dh0: Gradient w.r.t. initial hidden state
            dc0: Gradient w.r.t. initial cell state
        """
        self.cell.dW_f.fill(0)
        self.cell.dW_i.fill(0)
        self.cell.dW_c.fill(0)
        self.cell.dW_o.fill(0)
        self.cell.db_f.fill(0)
        self.cell.db_i.fill(0)
        self.cell.db_c.fill(0)
        self.cell.db_o.fill(0)

        batch_size, seq_len, _ = grad_outputs.shape
        grad_input = np.zeros((batch_size, seq_len, self.input_size), dtype=np.float32)

        if grad_h_last is None:
            dh_next = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            dh_next = grad_h_last
        if grad_c_last is None:
            dc_next = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            dc_next = grad_c_last

        # Retrieve input sequence
        for t in reversed(range(seq_len)):
            dh = grad_outputs[:, t, :] + dh_next
            self.cell.cache = self.cache[t]  
            dx_t, dh_prev, dc_prev = self.cell.backward(dh, dc_next)
            grad_input[:, t, :] = dx_t
            dh_next, dc_next = dh_prev, dc_prev
        dh0 = dh_next
        dc0 = dc_next
        return grad_input, dh0, dc0
