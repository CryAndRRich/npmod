from typing import List
import numpy as np
from ..layers import Layer
from ..optimizers import Optimizer, AdamW

class LAMB(AdamW):
    """
    Layer-wise Adaptive Moments optimizer for Batch training (LAMB) optimizer
    """
    def __init__(self,
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-6,
                 weight_decay: float = 0.01):
        super().__init__(network, learn_rate, beta1, beta2, epsilon, weight_decay)

    def step(self) -> None:
        self.timestep += 1
        for layer, moments, velocities in zip(self.net, self.moments, self.velocities):
            for param, grad, moment, velocity in zip(layer.parameters(), layer.gradients(), moments, velocities):
                grad += self.weight_decay * param
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                velocity[:] = self.beta2 * velocity + (1 - self.beta2) * (grad ** 2)

                m_hat = moment / (1 - self.beta1 ** self.timestep)
                v_hat = velocity / (1 - self.beta2 ** self.timestep)
                update = m_hat / (np.sqrt(v_hat) + self.epsilon)

                # LAMB normalization (trust ratio)
                w_norm = np.linalg.norm(param)
                u_norm = np.linalg.norm(update)
                trust_ratio = w_norm / u_norm if (w_norm > 0 and u_norm > 0) else 1.0

                param -= self.learn_rate * trust_ratio * update

class NovoGrad(Optimizer):
    """
    NovoGrad optimizer
    """
    def __init__(self,
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta1: float = 0.95,
                 beta2: float = 0.98,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(network, learn_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.moments = self._initialize_buffers()
        self.norm_squares = [0.0 for _ in self.net]

    def step(self) -> None:
        for i, (layer, moments) in enumerate(zip(self.net, self.moments)):
            grad_norm = 0.0
            # Compute global gradient norm per layer
            for g in layer.gradients():
                grad_norm += np.sum(g ** 2)
            grad_norm = np.sqrt(grad_norm) + self.epsilon

            # Update moving average of squared grad norms
            self.norm_squares[i] = self.beta2 * self.norm_squares[i] + (1 - self.beta2) * (grad_norm ** 2)
            denom = np.sqrt(self.norm_squares[i]) + self.epsilon

            # Parameter updates
            for param, grad, moment in zip(layer.parameters(), layer.gradients(), moments):
                g = grad / denom
                moment[:] = self.beta1 * moment + g + self.weight_decay * param
                param -= self.learn_rate * moment

class Ranger(Optimizer):
    """
    Ranger optimizer
    """
    def __init__(self,
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 k: int = 6,
                 alpha: float = 0.5):
        super().__init__(network, learn_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self.moments = self._initialize_buffers()
        self.velocities = self._initialize_buffers()
        self.k = k
        self.alpha = alpha
        self.slow_weights = [[p.copy() for p in layer.parameters()] for layer in self.net]
        self.step_counter = 0

    def step(self) -> None:
        self.timestep += 1
        self.step_counter += 1

        for layer, moments, velocities, slow in zip(self.net, self.moments, self.velocities, self.slow_weights):
            for param, grad, moment, velocity, slow_param in zip(layer.parameters(), layer.gradients(), moments, velocities, slow):
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                velocity[:] = self.beta2 * velocity + (1 - self.beta2) * (grad ** 2)

                # Compute rectified learning rate (RAdam)
                beta2_t = self.beta2 ** self.timestep
                rho_inf = 2 / (1 - self.beta2) - 1
                rho_t = rho_inf - 2 * self.timestep * beta2_t / (1 - beta2_t)
                if rho_t > 4:
                    r_t = np.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) /
                                  ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    step_size = self.learn_rate * r_t / (np.sqrt(velocity) + self.epsilon)
                else:
                    step_size = self.learn_rate

                param -= step_size * moment

                # Lookahead update every k steps
                if self.step_counter % self.k == 0:
                    slow_param += self.alpha * (param - slow_param)
                    param[:] = slow_param

class Apollo(Optimizer):
    """
    Apollo Optimizer
    """
    def __init__(self,
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        super().__init__(network, learn_rate)
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.velocities = self._initialize_buffers()   
        self.directions = self._initialize_buffers()

    def step(self) -> None:
        for layer, velocities, directions in zip(self.net, self.velocities, self.directions):
            for param, grad, velocity, direction in zip(layer.parameters(), layer.gradients(), velocities, directions):
                if self.weight_decay != 0:
                    grad += self.weight_decay * param

                velocity[:] = self.beta2 * velocity + (1 - self.beta2) * (grad ** 2)
                denom = np.sqrt(velocity) + self.epsilon

                # Compute quasi-Newton direction
                delta = grad / denom
                direction[:] = self.beta * direction + (1 - self.beta) * delta

                # Trust ratio (Apollo specific)
                w_norm = np.linalg.norm(param)
                d_norm = np.linalg.norm(direction)
                trust_ratio = w_norm / d_norm if (w_norm > 0 and d_norm > 0) else 1.0

                param -= self.learn_rate * trust_ratio * direction

class Sophia(Optimizer):
    """
    Sophia optimizer
    """
    def __init__(self, 
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta1: float = 0.965,
                 beta2: float = 0.99,
                 rho: float = 0.04,
                 epsilon: float = 1e-8) -> None:
        """
        Parameters:
            network: The neural network layers to optimize
            learn_rate: Learning rate
            beta1, beta2: Exponential moving average factors
            rho: Scaling for Hessian approximation (used for stability)
            epsilon: Small constant for numerical stability
        """
        super().__init__(network, learn_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.epsilon = epsilon
        self.moments = self._initialize_buffers()
        self.hessians = self._initialize_buffers()  # approximate Hessian diagonal
        self.timestep = 0

    def step(self) -> None:
        self.timestep += 1
        for layer, moments, hessians in zip(self.net, self.moments, self.hessians):
            for param, grad, moment, hessian in zip(layer.parameters(), layer.gradients(), moments, hessians):
                # EMA of gradients and Hessian approximation
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                hessian[:] = self.beta2 * hessian + (1 - self.beta2) * (grad ** 2)

                # Bias correction
                m_hat = moment / (1 - self.beta1 ** self.timestep)
                h_hat = hessian / (1 - self.beta2 ** self.timestep)

                # Sophia update rule
                denom = np.sqrt(h_hat) + self.rho + self.epsilon
                param -= self.learn_rate * m_hat / denom

class Lion(Optimizer):
    """
    Lion optimizer
    """
    def __init__(self, 
                 network: List[Layer],
                 learn_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.99):
        """
        Initializes the Lion optimizer

        Parameters:
            network: The neural network layers to be optimized
            learn_rate: The learning rate for parameter updates
            beta1: The decay rate for the first moment estimates
            beta2: The decay rate for the second moment estimates
        """
        super().__init__(network, learn_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.moments = self._initialize_buffers()

    def step(self) -> None:
        for layer, moments in zip(self.net, self.moments):
            for param, grad, moment in zip(layer.parameters(), layer.gradients(), moments):
                moment[:] = self.beta1 * moment + (1 - self.beta1) * grad
                update = np.sign(self.beta1 * moment + (1 - self.beta1) * grad)
                param -= self.learn_rate * update