from typing import List
import numpy as np

from .layers import Layer
from .losses import Loss

class Container():
    """
    Contains many Layers
    """
    def __init__(self, 
                 is_eval: bool = False,
                 layers: List[Layer] = None) -> None:
        """
        Parameters:
            is_eval: Whether the Container is in eval mode
            layers: Layer to be included in the container
        """
        self.is_eval = is_eval
        self.layers = layers if layers is not None else []

    def forward(self, inputs):
        """
        Forward pass of Sequential

        Parameters:
            inputs: Inputs to the Layer
        """
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def get_layers(self) -> List[Layer]:
        return self.layers
    
    def add_layer(self, layer: Layer) -> None:
        """
        Adds a new layer to the container
        """
        self.layers.append(layer)
    
    def train(self) -> None:
        self.is_eval = False
        for layer in self.layers:
            layer.training = True

    def eval(self) -> None:
        self.is_eval = True
        for layer in self.layers:
            layer.training = False

class Sequential(Container):
    """
    Sequential Container. Outputs of one layer are passed as inputs to the next layer, sequentially
    """
    def __init__(self, 
                 eval: bool = False,
                 layers: List[Layer] = None) -> None:
        super().__init__(eval, layers)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def backward(self, loss: Loss) -> np.ndarray:
        grad = loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad