from typing import List
from .layers import Layer
from .losses import Loss

class Container:
    """
    Contains many Layers
    """
    def __init__(self, 
                 eval: bool = False,
                 layers: List[Layer] = None):
        """
        --------------------------------------------------
        Parameters:
            eval: Whether the Container is in eval mode
            layers: Layer to be included in the container
        """
        self.eval = eval
        self.layers = layers

    def forward(self, inputs):
        """
        Forward pass of Sequential

        --------------------------------------------------
        Parameters:
            inputs: Inputs to the Layer
        """
        pass

    def __call__(self, inputs):
        """
        Calls the forward method

        --------------------------------------------------
        Parameters:
            inputs: Inputs to the container
        """
        return self.forward(inputs)

    def get_layers(self) -> List[Layer]:
        return self.layers

class Sequential(Container):
    """
    Sequential Container.
    Outputs of one layer are passed as inputs to the next layer, sequentially
    """
    def __init__(self, 
                 eval: bool = False,
                 layers: List[Layer] = None):
        super().__init__(eval, layers)
    
    def forward(self, inputs):
        for layer in self.layers:
            output = layer(inputs)
            inputs = output
        return output
    
    def backward(self, loss: Loss):
        grad = loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)