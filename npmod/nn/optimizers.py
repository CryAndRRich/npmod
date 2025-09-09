from typing import List
import numpy as np
from .layers import Layer

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

class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm (AdaGrad) optimizer
    """
    def __init__(self, 
                 network: List[Layer], 
                 learn_rate: float = 0.01, 
                 epsilon: float = 1e-10) -> None:
        """
        Initializes the AdaGrad optimizer

        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
            epsilon: A small constant to prevent division by zero
        """
        super().__init__(network, learn_rate)
        self.epsilon = epsilon
        self.squares = self._initialize_buffers()
    
    def step(self) -> None:
        for layer, squares in zip(self.net, self.squares):
            for param, grad, square in zip(layer.parameters(), layer.gradients(), squares):
                square[:] += grad ** 2
                param -= self.learn_rate * grad / (np.sqrt(square) + self.epsilon)

class RMSprop(Optimizer):
    """
    RMSprop optimizer
    """
    def __init__(self, 
                 network: List[Layer], 
                 learn_rate: float = 0.01,
                 beta: float = 0.9, 
                 epsilon: float = 1e-8) -> None:
        """
        Initializes the RMSprop optimizer

        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
            beta: The decay rate for moving average of squared gradients
            epsilon: A small constant to prevent division by zero
        """
        super().__init__(network, learn_rate)
        self.epsilon = epsilon
        self.beta = beta
        self.squares = self._initialize_buffers()
    
    def step(self) -> None:
        for layer, squares in zip(self.net, self.squares):
            for param, grad, square in zip(layer.parameters(), layer.gradients(), squares):
                square[:] = self.beta * square + (1 - self.beta) * grad ** 2
                param -= self.learn_rate * grad / (np.sqrt(square) + self.epsilon)

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer
    """
    def __init__(self, 
                 network: List[Layer], 
                 learn_rate: float = 0.001, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon: float = 1e-8) -> None:
        """
        Initializes the Adam optimizer

        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
            beta1: The decay rate for the first moment estimates
            beta2: The decay rate for the second moment estimates
            epsilon: A small constant to prevent division by zero
        """
        super().__init__(network, learn_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self.moments = self._initialize_buffers()
        self.velocities = self._initialize_buffers()
    
    def step(self) -> None:
        self.timestep += 1
        for layer, moments, velocities in zip(self.net, self.moments, self.velocities):
            for param, grad, moment, velocity in zip(layer.parameters(), layer.gradients(), moments, velocities):
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                velocity[:] = self.beta2 * velocity + (1 - self.beta2) * (grad ** 2)

                m_hat = moment / (1 - self.beta1 ** self.timestep)
                v_hat = velocity / (1 - self.beta2 ** self.timestep)

                param -= self.learn_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
