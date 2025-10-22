import numpy as np
from ..losses import Loss

class KLDiv(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> np.ndarray:
        super().forward(input, target)
        eps = 1e-9
        self.input_log = np.log(input + eps)  
        self.target = target
        loss = np.mean(np.sum(target * (np.log(target + eps) - self.input_log), axis=-1))
        return loss
    
    def backward(self) -> np.ndarray:
        self.gradient = -self.target / self.target.shape[0]
        return self.gradient
    
    def __str__(self) -> str:
        return "Kullback-Leibler Divergence Loss (KLDiv)"

class JSDiv(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> float:
        super().forward(input, target)
        eps = 1e-9
        m = 0.5 * (input + target)
        kl1 = np.sum(target * np.log((target + eps) / (m + eps)), axis=-1)
        kl2 = np.sum(input * np.log((input + eps) / (m + eps)), axis=-1)
        loss = 0.5 * np.mean(kl1 + kl2)
        return loss

    def backward(self) -> np.ndarray:
        eps = 1e-9
        m = 0.5 * (self.input + self.target)
        grad = 0.5 * (np.log((self.input + eps) / (m + eps)) + 1 - (self.target + eps) / (m + eps))
        self.gradient = grad / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Jensen-Shannon Divergence (JSDiv)"

class Wasserstein(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> float:
        super().forward(input, target)
        loss = -np.mean(input * target)
        return loss

    def backward(self) -> np.ndarray:
        self.gradient = -self.target / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Wasserstein Loss"