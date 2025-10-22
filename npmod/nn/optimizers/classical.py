from typing import List
from ..layers import Layer
from ..optimizers import Optimizer

class GD(Optimizer):
    """
    Gradient Descent optimizer
    """
    def step(self) -> None:
        for layer in self.net:
            for param, grad in zip(layer.parameters(), layer.gradients()):
                param -= self.learn_rate * grad

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum
    """
    def __init__(self, 
                 network: List[Layer],
                 learn_rate: float = 0.01,
                 momentum: float = 0.01) -> None:
        """
        Initializes the SGD optimizer with momentum

        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
            momentum: The momentum factor to accelerate updates
        """
        super().__init__(network, learn_rate)
        self.momentum = momentum
        self.velocities = self._initialize_buffers()
    
    def step(self) -> None:
        for layer, velocities in zip(self.net, self.velocities):
            for param, grad, velocity in zip(layer.parameters(), layer.gradients(), velocities):
                velocity[:] = self.momentum * velocity + grad
                param -= self.learn_rate * velocity