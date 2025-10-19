import numpy as np
from ..layers import Layer

class WeightNorm(Layer):
    """
    Weight Normalization wrapper
    """
    def __init__(self, 
                 layer: Layer, 
                 eps: float = 1e-8) -> None:
        """
        Parameters:
            layer: layer to wrap (must have 'weight' and 'weight_grad')
            eps: small constant for numerical stability
        """
        super().__init__()
        assert hasattr(layer, "weight"), "Layer must have attribute 'weight'"
        assert hasattr(layer, "weight_grad"), "Layer must have attribute 'weight_grad'"

        self.layer = layer
        self.eps = eps

        w = self.layer.weight
        if w.ndim == 2:
            # Linear
            self.dim = 1
        else:
            # Conv
            self.dim = tuple(range(1, w.ndim))  

        # Initialize v and g
        self.v = w.copy()
        self.g = np.linalg.norm(w, axis=self.dim, keepdims=True)

        # Initialize gradients
        self.v_grad = np.zeros_like(self.v)
        self.g_grad = np.zeros_like(self.g)

        # Set initial weight
        self.layer.weight = self.compute_weight()

    def parameters(self):
        yield from self.layer.parameters()
        yield self.v
        yield self.g

    def gradients(self):
        yield from self.layer.gradients()
        yield self.v_grad
        yield self.g_grad

    def compute_weight(self) -> np.ndarray:
        """Calculate weight from g and v"""
        v_norm = np.linalg.norm(self.v, axis=self.dim, keepdims=True) + self.eps
        w = self.g * self.v / v_norm
        return w.astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.layer.weight = self.compute_weight()
        return self.layer.forward(x)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        grad_input = self.layer.backward(previous_grad)

        grad_w = getattr(self.layer, "weight_grad", None)
        if grad_w is None:
            raise ValueError("Wrapped layer must compute 'weight_grad'")

        v = self.v
        g = self.g
        eps = self.eps
        dim = self.dim

        v_norm = np.linalg.norm(v, axis=dim, keepdims=True) + eps
        v_unit = v / v_norm

        self.g_grad = np.sum(grad_w * v_unit, axis=dim, keepdims=True)

        self.v_grad = (g / v_norm) * (grad_w - np.sum(grad_w * v_unit, axis=dim, keepdims=True) * v_unit)

        return grad_input
