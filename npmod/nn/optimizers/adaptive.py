from typing import List
import numpy as np
from ..layers import Layer
from ..optimizers import Optimizer

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

class AdamW(Adam):
    """
    Adam with decoupled weight decay (AdamW) optimizer
    """
    def __init__(self, 
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01) -> None:
        super().__init__(network, learn_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay # Decoupled weight decay factor

    def step(self) -> None:
        self.timestep += 1
        for layer, moments, velocities in zip(self.net, self.moments, self.velocities):
            for param, grad, moment, velocity in zip(layer.parameters(), layer.gradients(), moments, velocities):
                grad += self.weight_decay * param
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                velocity[:] = self.beta2 * velocity + (1 - self.beta2) * grad ** 2
                m_hat = moment / (1 - self.beta1 ** self.timestep)
                v_hat = velocity / (1 - self.beta2 ** self.timestep)
                param -= self.learn_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RAdam(Adam):
    """
    Rectified Adam (RAdam) optimizer
    """
    def __init__(self,
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        super().__init__(network, learn_rate, beta1, beta2, epsilon)
        # Constants for rectification
        self.rho_inf = 2 / (1 - self.beta2) - 1

    def step(self) -> None:
        self.timestep += 1
        for layer, moments, velocities in zip(self.net, self.moments, self.velocities):
            for param, grad, moment, velocity in zip(layer.parameters(), layer.gradients(), moments, velocities):
                # Update biased first & second moments
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                velocity[:] = self.beta2 * velocity + (1 - self.beta2) * (grad ** 2)

                # Compute bias correction
                beta1_t = self.beta1 ** self.timestep
                beta2_t = self.beta2 ** self.timestep

                # Compute rectification term
                rho_t = self.rho_inf - 2 * self.timestep * beta2_t / (1 - beta2_t)

                if rho_t > 4:
                    # Rectified update
                    r_t = np.sqrt(
                        ((rho_t - 4) * (rho_t - 2) * self.rho_inf) /
                        ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t)
                    )
                    m_hat = moment / (1 - beta1_t)
                    param -= self.learn_rate * r_t * m_hat / (np.sqrt(velocity) + self.epsilon)
                else:
                    # Unrectified update
                    m_hat = moment / (1 - beta1_t)
                    param -= self.learn_rate * m_hat

class AdaBelief(Adam):
    """
    AdaBelief optimizer
    """
    def __init__(self, 
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        super().__init__(network, learn_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay

    def step(self) -> None:
        self.timestep += 1
        for layer, moments, velocities in zip(self.net, self.moments, self.velocities):
            for param, grad, moment, velocity in zip(layer.parameters(), layer.gradients(), moments, velocities):
                if self.weight_decay != 0:
                    grad += self.weight_decay * param
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                velocity[:] = self.beta2 * velocity + (1 - self.beta2) * ((grad - moment) ** 2)
                m_hat = moment / (1 - self.beta1 ** self.timestep)
                v_hat = velocity / (1 - self.beta2 ** self.timestep)
                param -= self.learn_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class Lookahead(Optimizer):
    """
    Lookahead optimizer
    """
    def __init__(self, 
                 network: List[Layer],
                 inner_optimizer: Optimizer,
                 k: int = 5,
                 alpha: float = 0.5) -> None:
        """
        Parameters:
            network: The neural network layers to optimize
            inner_optimizer: The inner optimizer (e.g., Adam, SGD)
            k: Number of inner steps before slow update
            alpha: Interpolation factor for slow weights
        """
        super().__init__(network, inner_optimizer.learn_rate)
        self.inner_optimizer = inner_optimizer
        self.k = k
        self.alpha = alpha
        # Create slow weights as a copy of current parameters
        self.slow_weights = [[p.copy() for p in layer.parameters()] for layer in self.net]
        self.step_counter = 0

    def step(self) -> None:
        # Perform inner optimizer step
        self.inner_optimizer.step()
        self.step_counter += 1

        # Every k steps, perform the slow update
        if self.step_counter % self.k == 0:
            for layer, slow_params in zip(self.net, self.slow_weights):
                for p, sp in zip(layer.parameters(), slow_params):
                    sp += self.alpha * (p - sp)
                    p[:] = sp  # sync fast weights back to slow