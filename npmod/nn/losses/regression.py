import numpy as np
from ..losses import Loss

class MAE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> np.ndarray:
        super().forward(input, target)
        loss = np.mean(np.abs(self.input - self.target))
        return loss

    def backward(self) -> np.ndarray:
        self.gradient = np.sign(self.input - self.target) / self.input.shape[0]
        return self.gradient

    def __str__(self) -> str:
        return "Mean Absolute Error (MAE)"

class MSE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> np.ndarray:
        super().forward(input, target)
        loss = np.mean((self.input - self.target) ** 2)
        return loss

    def backward(self) -> np.ndarray:
        self.gradient = 2 * (self.input - self.target) / self.input.shape[0]
        return self.gradient

    def __str__(self) -> str:
        return "Mean Squared Error (MSE)"

class MALE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> np.ndarray:
        assert np.any(input > 0) and np.any(target > 0), \
            "Input and target must be greater than 0 for log transformation"
        
        super().forward(input, target)
        loss = np.mean((np.log(self.input) - np.log(self.target)) ** 2)
        return loss

    def backward(self) -> np.ndarray:
        assert np.any(self.input > 0) and np.any(self.target > 0), \
            "Input and target must be greater than 0 for log transformation"
        
        self.gradient = 2 * (np.log(self.input) - np.log(self.target)) / (self.input * self.input.shape[0])
        return self.gradient

    def __str__(self) -> str:
        return "Mean Absolute Log Error (MALE)"

class RSquared(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> np.ndarray:
        super().forward(input, target)
        mean_target = np.mean(target)
        ss_res = np.sum((target - input) ** 2)
        ss_tot = np.sum((target - mean_target) ** 2)
        if ss_tot == 0:
            raise ValueError("The denominator is zero, cannot compute R^2")
        self.ss_tot = ss_tot  # cache for backward
        loss = ss_res / ss_tot
        return loss

    def backward(self) -> np.ndarray:
        self.gradient = -2 * (self.target - self.input) / self.ss_tot
        return self.gradient
    
    def __str__(self) -> str:
        return "R Squared (R2)"

class MAPE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> np.ndarray:
        super().forward(input, target)
        self.eps = 1e-9
        self.loss = np.mean(np.abs(input - target) / (target + self.eps))
        return self.loss

    def backward(self) -> np.ndarray:
        N = self.input.shape[0]
        self.gradient = np.sign(self.input - self.target) / ((self.target + self.eps) * N)
        return self.gradient

    def __str__(self) -> str:
        return "Mean Absolute Percentage Error (MAPE)"

class wMAPE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                weights: float | np.ndarray = None) -> float:

        if weights is None:
            weights = np.ones(input.shape[0], dtype=np.float32)

        self.weights = weights.astype(np.float32) if isinstance(weights, np.ndarray) else np.full(input.shape[0], weights, dtype=np.float32)
        super().forward(input, target)
        eps = 1e-9
        numerator = np.sum(np.abs(self.target - self.input) * self.weights)
        denominator = np.sum(np.abs(self.target) * self.weights) + eps
        self.denominator = denominator  # cache for backward
        self.numerator = numerator
        loss = numerator / denominator
        return loss

    def backward(self) -> np.ndarray:
        self.gradient = self.weights * np.sign(self.input - self.target) / self.denominator
        return self.gradient

    def __str__(self) -> str:
        return "Weighted Mean Absolute Percentage Error (wMAPE)"

class SmoothL1(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                beta: float = 1.0) -> float:
        super().forward(input, target)
        self.beta = beta
        ln = np.abs(input - target)
        loss = np.mean(np.where(ln < self.beta, 0.5 * (ln ** 2) / self.beta, ln - 0.5 * self.beta))
        return loss
    
    def backward(self) -> np.ndarray:
        ln = self.input - self.target
        self.gradient = np.where(np.abs(ln) < self.beta, ln, np.sign(ln)) / self.input.shape[0]
        return self.gradient

    def __str__(self) -> str:
        return "Smooth L1 Loss (SmoothL1)"

class Huber(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray,
                delta: float = 1.0) -> float:
        super().forward(input, target)
        self.delta = delta
        diff = np.abs(input - target)
        loss = np.mean(np.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta)))
        return loss

    def backward(self) -> np.ndarray:
        diff = self.input - self.target
        grad = np.where(np.abs(diff) <= self.delta, diff, self.delta * np.sign(diff))
        self.gradient = grad / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Huber Loss"

class LogCosh(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> float:
        super().forward(input, target)
        loss = np.mean(np.log(np.cosh(input - target)))
        return loss

    def backward(self) -> np.ndarray:
        diff = self.input - self.target
        self.gradient = np.tanh(diff) / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Log-Cosh Loss"

class QuantileLoss(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                quantile: float = 0.5) -> float:
        super().forward(input, target)
        self.q = quantile
        diff = target - input
        loss = np.mean(np.maximum(self.q * diff, (self.q - 1) * diff))
        return loss

    def backward(self) -> np.ndarray:
        diff = self.target - self.input
        self.gradient = np.where(diff >= 0, -self.q, 1 - self.q) / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Quantile Loss"
