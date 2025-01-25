from typing import List
import numpy as np

class Layer:
    """
    Fundamental building block of the model
    """
    def __init__(self):
        self.training = True

    def parameters(self):
        """
        Returns the parameters of the layer
        """
        pass

    def gradients(self):
        """
        Returns the gradients of the parameters
        """
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer
        """
        pass

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer
        """
        return previous_grad
    
    def __call__(self, x):
        return self.forward(x)

class Linear(Layer):
    """
    Fully connected linear layer
    """
    def __init__(self, in_features: int, out_features: int):
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

    def parameters(self) -> List[np.ndarray, np.ndarray]:
        return [self.weight, self.bias]

    def gradients(self) -> List[np.ndarray, np.ndarray]:
        return [self.weight_grad, self.bias_grad]

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2 or len(x.shape) == 3, "Input must be 2D or 3D."

        self.inputs = x.copy()
        input_shape = self.inputs.shape

        if len(input_shape) == 3:
            b, c, _ = input_shape
            output_shape = (b, c, self.out_features)
        else:
            b, _ = input_shape
            output_shape = (b, self.out_features)

        return (np.dot(self.inputs.reshape(-1, self.in_features), self.weight.T)
                .reshape(output_shape) + self.bias)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        assert len(previous_grad.shape) == 2 or len(previous_grad.shape) == 3, \
            "Gradient must be 2D or 3D"

        if len(previous_grad.shape) == 3:
            sum_axis = (0, 1)
        else:
            sum_axis = (0,)

        batch_size = previous_grad.shape[0]
        grad_input = (np.dot(previous_grad.reshape(-1, self.out_features), self.weight)
                      .reshape(self.inputs.shape))

        self.weight_grad = (np.dot(previous_grad.reshape(-1, self.out_features).T, 
                                   self.inputs.reshape(batch_size, -1)).reshape(self.weight_grad.shape))
        self.bias_grad = np.sum(previous_grad, axis=sum_axis)

        return grad_input

class Dropout(Layer):
    """
    Dropout layer for regularization
    """
    def __init__(self, prob: float):
        """
        Parameters:
            prob: Probability with which to keep the inputs
        """
        super().__init__()
        assert 0 < prob <= 1, "Probability must be between 0 and 1"
        self.prob = prob
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.binomial(1, self.prob, x.shape).astype(np.float32)
            return x * self.mask / self.prob
        else:
            return x

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        return previous_grad * self.mask / self.prob if self.training else previous_grad

class Flatten(Layer):
    """
    Flatten layer to reshape input tensors
    """
    def __init__(self, keep_channels: bool = False):
        """
        Parameters:
            keep_channels: If True, retains the channel dimension during flattening.\
                           Otherwise, flattens all dimensions except the batch size
        """
        super().__init__()
        self.keep_channels = keep_channels
        self.forward_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.forward_shape = x.shape
        if self.keep_channels:
            return x.reshape(x.shape[0], x.shape[1], -1)
        else:
            return x.reshape(x.shape[0], -1)

    def backward(self, previous_grad: np.ndarray) -> np.ndarray:
        return previous_grad.reshape(self.forward_shape)
