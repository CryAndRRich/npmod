from typing import List
import numpy as np
from ..layers import Layer

class Optimizer():
    """
    Base class for optimizers
    """
    def __init__(self, 
                 network: List[Layer],
                 learn_rate: float = 0.01) -> None:
        """
        Initializes the optimizer with the given network and learning rate

        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
        """
        self.net = network
        self.learn_rate = learn_rate

    def _initialize_buffers(self) -> List[List[np.ndarray]]:
        """
        Initializes buffers (e.grad., velocities, moments) for each layer parameters

        Returns:
            List[List[np.ndarray]]: A list of buffers initialized to zero, with the same shape as the network parameters
        """
        return [[np.zeros_like(p) for p in layer.parameters()] for layer in self.net]
    
    def step(self) -> None:
        """
        Performs a single optimization step
        """
        pass

from .classical import GD, SGD
from .adaptive import AdaGrad, RMSprop, Adam, AdamW, RAdam, AdaBelief, Lookahead
from .modern import LAMB, NovoGrad, Ranger, Apollo, Sophia, Lion