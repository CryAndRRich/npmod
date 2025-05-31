import numpy as np
from ..layers import Layer

class Embedding(Layer):
    """
    Embedding layer for representing discrete tokens as dense vectors
    """
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int) -> None:
        """
        Parameters:
            num_embeddings: Number of unique tokens
            embedding_dim: Dimension of each embedding vector
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = (np.random.randn(num_embeddings, embedding_dim) * 0.01).astype(np.float32)
        self.grad_weight = np.zeros_like(self.weight, dtype=np.float32)
        self.input_indices = None

    def parameters(self):
        yield self.weight

    def gradients(self):
        yield self.grad_weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_indices = x
        return self.weight[x]

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        self.grad_weight.fill(0)
        batch_size, seq_len, _ = previous_grad.shape
        for b in range(batch_size):
            for s in range(seq_len):
                idx = self.input_indices[b, s]
                self.grad_weight[idx] += previous_grad[b, s]
        return None  