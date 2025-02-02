from typing import List
import numpy as np
from .layers import Layer

class Optimizer():
    """
    Base class for optimizers
    """
    def __init__(self, 
                 network: List[Layer],
                 learn_rate: float = 0.01):
        """
        Initializes the optimizer with the given network and learning rate

        --------------------------------------------------
        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
        """
        self.net = network
        self.learn_rate = learn_rate

    def _initialize_buffers(self) -> List[List[np.ndarray]]:
        """
        Initializes buffers (e.g., velocities, moments) for each layer's parameters

        --------------------------------------------------
        Returns:
            List[List[np.ndarray]]: A list of buffers initialized to zero, with the same shape as the network parameters
        """
        return [[np.zeros_like(param) for param in layer.parameters()] for layer in self.net]

    def step(self):
        """
        Performs a single optimization step. To be implemented by subclasses
        """
        pass

class GD(Optimizer):
    """
    Gradient Descent optimizer
    """
    def __init__(self, 
                 network: List[Layer],
                 learn_rate: float = 0.01):
        """
        Initializes the Gradient Descent optimizer

        --------------------------------------------------
        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
        """
        super().__init__(network, learn_rate)
    
    def step(self):
        """
        Performs a single optimization step using gradient descent
        Updates parameters by subtracting the gradient scaled by the learning rate
        """
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
                 momentum: float = 0.01):
        """
        Initializes the SGD optimizer with momentum

        --------------------------------------------------
        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
            momentum: The momentum factor to accelerate updates
        """
        super().__init__(network, learn_rate)
        self.momentum = momentum
        self.velocities = self._initialize_buffers()
    
    def step(self):
        """
        Performs a single optimization step using SGD with momentum
        Updates parameters using a combination of gradient and momentum
        """
        for layer, velocities in zip(self.net, self.velocities):
            for param, grad, velocity in zip(layer.parameters(), layer.gradients(), velocities):
                velocity *= self.momentum
                velocity -= self.learn_rate * grad
                param += velocity

class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm (AdaGrad) optimizer
    """
    def __init__(self, 
                 network: List[Layer], 
                 learn_rate: float = 0.01, 
                 epsilon: float = 1e-8):
        """
        Initializes the AdaGrad optimizer

        --------------------------------------------------
        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
            epsilon: A small constant to prevent division by zero
        """
        super().__init__(network, learn_rate)
        self.epsilon = epsilon
        self.velocities = self._initialize_buffers()
    
    def step(self):
        """
        Performs a single optimization step using AdaGrad
        Adapts the learning rate for each parameter based on historical gradients
        """
        for layer, velocities in zip(self.net, self.velocities):
            for param, grad, velocity in zip(layer.parameters(), layer.gradients(), velocities):
                velocity += grad ** 2
                rate_change = np.sqrt(velocity) + self.epsilon
                adapted_learning_rate = self.learn_rate / rate_change
                param -= adapted_learning_rate * grad

class RMSprop(Optimizer):
    """
    RMSprop optimizer
    """
    def __init__(self, 
                 network: List[Layer], 
                 learn_rate: float = 0.01,
                 beta: float = 0.9, 
                 epsilon: float = 1e-8):
        """
        Initializes the RMSprop optimizer

        --------------------------------------------------
        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
            beta: The decay rate for moving average of squared gradients
            epsilon: A small constant to prevent division by zero
        """
        super().__init__(network, learn_rate)
        self.epsilon = epsilon
        self.beta = beta
        self.velocities = self._initialize_buffers()
    
    def step(self):
        """
        Performs a single optimization step using RMSprop
        Uses a moving average of squared gradients to scale the learning rate
        """
        for layer, velocities in zip(self.net, self.velocities):
            for param, grad, velocity in zip(layer.parameters(), layer.gradients(), velocities):
                velocity[:] = self.beta * velocity + (1 - self.beta) * (grad ** 2)
                param -= self.learn_rate * grad / (np.sqrt(velocity) + self.epsilon)

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer
    """
    def __init__(self, 
                 network: List[Layer], 
                 learn_rate: float = 0.001, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon: float = 1e-8):
        """
        Initializes the Adam optimizer

        --------------------------------------------------
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
    
    def step(self):
        """
        Performs a single optimization step using Adam
        Combines the benefits of momentum and RMSprop for parameter updates
        """
        self.timestep += 1
        for layer, moments, velocities in zip(self.net, self.moments, self.velocities):
            for param, grad, moment, velocity in zip(layer.parameters(), layer.gradients(), moments, velocities):
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                velocity[:] = self.beta2 * velocity + (1 - self.beta2) * (grad ** 2)

                corrected_moment = moment / (1 - self.beta1 ** self.timestep)
                corrected_velocity = velocity / (1 - self.beta2 ** self.timestep)

                param -= self.learn_rate * corrected_moment / (np.sqrt(corrected_velocity) + self.epsilon)
