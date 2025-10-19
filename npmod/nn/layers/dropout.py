import numpy as np
from ..layers import Layer

class Dropout(Layer):
    """
    Dropout layer for regularization
    """
    def __init__(self, keep_prob: float) ->None:
        """
        Parameters:
            keep_prob: Probability with which to keep the inputs
        """
        super().__init__()
        assert 0 < keep_prob <= 1, "Probability must be between 0 and 1"
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.binomial(1, self.keep_prob, x.shape).astype(np.float32)
            return (x * self.mask) / self.keep_prob
        else:
            return x

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        if self.training:
            return (previous_grad * self.mask) / self.keep_prob
        else:
            return previous_grad

class DropConnect(Layer):
    """
    DropConnect layer for regularization
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 keep_prob: float) -> None:
        """
        Parameters:
            in_features: Number of input features
            out_features: Number of output features
            keep_prob: Probability with which to keep a connection
        """
        super().__init__()
        assert 0 < keep_prob <= 1, "keep_prob must be between 0 and 1"
        self.keep_prob = keep_prob
        
        # Initialize weights and bias
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((out_features, 1))
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache
        self.mask = None
        self.x = None

    def parameters(self):
        return [self.W, self.b]

    def gradients(self):
        return [self.dW, self.db]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        if self.training:
            # Apply random mask to weights
            self.mask = np.random.binomial(1, self.keep_prob, self.W.shape).astype(np.float32)
            W_dropped = (self.W * self.mask) / self.keep_prob
        else:
            # Use full weights during inference
            W_dropped = self.W

        out = np.dot(W_dropped, x) + self.b
        return out

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        if self.training:
            W_used = (self.W * self.mask) / self.keep_prob
        else:
            W_used = self.W

        self.dW = np.dot(previous_grad, self.x.T)
        self.db = np.sum(previous_grad, axis=1, keepdims=True)

        dx = np.dot(W_used.T, previous_grad)
        return dx