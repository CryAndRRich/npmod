import numpy as np
from ..layers import Layer

class Linear(Layer):
    """
    Fully connected linear layer
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int) ->None:
        """
        Parameters:
            in_features: Number of inputs to the Layer
            out_features: Number of outputs from the Layer
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.inputs = None
        # Weights of the Layer
        self.weight = (np.random.randn(out_features, in_features) * 0.01).astype(np.float32)
        # Bias of the Layer
        self.bias = np.zeros(out_features, dtype=np.float32)

        self.weight_grad = np.zeros((out_features, in_features), dtype=np.float32)
        self.bias_grad = np.zeros(out_features, dtype=np.float32)

    def parameters(self):
        yield self.weight
        yield self.bias

    def gradients(self):
        yield self.weight_grad
        yield self.bias_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2 or len(x.shape) == 3, \
            "Input must be 2D or 3D."

        self.inputs = x
        input_shape = x.shape

        if len(input_shape) == 3:
            b, c, _ = input_shape
            output_shape = (b, c, self.out_features)
        else:
            b, _ = input_shape
            output_shape = (b, self.out_features)

        return np.dot(self.inputs.reshape(-1, self.in_features), 
                      self.weight.T).reshape(output_shape) + self.bias

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        assert len(previous_grad.shape) == 2 or len(previous_grad.shape) == 3, \
            "Gradient must be 2D or 3D"

        grad_input = np.dot(previous_grad.reshape(-1, self.out_features), self.weight) \
                  .reshape(self.inputs.shape)

        self.weight_grad = np.dot(previous_grad.reshape(-1, self.out_features).T,
                                self.inputs.reshape(-1, self.in_features))

        self.bias_grad = np.sum(previous_grad, axis=tuple(range(previous_grad.ndim - 1)))

        return grad_input