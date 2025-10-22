import numpy as np
from ..losses import Loss

class CE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> np.ndarray:
        super().forward(input, target)
        eps = 1e-9
        self.input_stable = input - np.max(input, axis=-1, keepdims=True)
        exp_input = np.exp(self.input_stable)
        self.softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)

        # Convert target to one-hot if necessary
        if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1):
            labels = target.reshape(-1).astype(np.int64)
            N = labels.shape[0]
            one_hot = np.zeros((N, input.shape[-1]), dtype=np.float32)
            one_hot[np.arange(N), labels] = 1.0
            self.target = one_hot
        else:
            self.target = target.astype(np.float32)

        loss = -np.mean(np.sum(self.target * np.log(self.softmax + eps), axis=-1))
        return float(loss)
    
    def backward(self) -> np.ndarray:
        self.gradient = (self.softmax - self.target) / self.input_stable.shape[0]
        return self.gradient
    
    def __str__(self) -> str:
        return "Cross Entropy Loss (CE)"

class BCE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                weights: float | np.ndarray = None) -> float:

        super().forward(input, target)
        eps = 1e-9

        if weights is None:
            weights = np.ones_like(input, dtype=np.float32)
        elif isinstance(weights, (float, int)):
            weights = np.full_like(input, weights, dtype=np.float32)
        else:
            weights = weights.astype(np.float32)
        
        self.weights = weights
        self.input = np.clip(input, eps, 1 - eps)  # avoid log(0)
        self.target = target
        
        # Binary cross entropy with optional weights
        loss = -np.mean(weights * (target * np.log(self.input) + (1 - target) * np.log(1 - self.input)))
        return loss
    
    def backward(self) -> np.ndarray:
        eps = 1e-9
        self.gradient = self.weights * (self.input - self.target) / (self.input * (1 - self.input) + eps) / self.input.shape[0]
        return self.gradient
    
    def __str__(self) -> str:
        return "Binary Cross Entropy (BCE)"
    
class FocalLoss(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                gamma: float = 2.0, 
                alpha: float = 0.25) -> float:
        super().forward(input, target)
        eps = 1e-9
        self.gamma = gamma
        self.alpha = alpha

        # Softmax
        self.input_stable = input - np.max(input, axis=-1, keepdims=True)
        exp_input = np.exp(self.input_stable)
        self.probs = exp_input / np.sum(exp_input, axis=-1, keepdims=True)

        # Convert to one-hot
        if target.ndim == 1:
            N = target.shape[0]
            one_hot = np.zeros_like(self.probs)
            one_hot[np.arange(N), target.astype(np.int64)] = 1.0
            self.target = one_hot
        else:
            self.target = target.astype(np.float32)

        pt = np.sum(self.target * self.probs, axis=-1) + eps
        loss = -np.mean(self.alpha * (1 - pt) ** self.gamma * np.log(pt))
        return float(loss)

    def backward(self) -> np.ndarray:
        eps = 1e-9
        pt = np.sum(self.target * self.probs, axis=-1, keepdims=True) + eps
        grad = self.alpha * ((self.gamma * (1 - pt) ** (self.gamma - 1) * pt * np.log(pt))
                             - (1 - pt) ** self.gamma) * (self.probs - self.target)
        self.gradient = grad / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Focal Loss"

class LabelSmoothingCE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                smoothing: float = 0.1) -> float:
        super().forward(input, target)
        eps = 1e-9
        self.smoothing = smoothing

        self.input_stable = input - np.max(input, axis=-1, keepdims=True)
        exp_input = np.exp(self.input_stable)
        self.softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)

        N, C = input.shape

        if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1):
            labels = target.reshape(-1).astype(np.int64)
            one_hot = np.zeros((N, C), dtype=np.float32)
            one_hot[np.arange(N), labels] = 1.0
            base_target = one_hot
        else:
            if target.shape != (N, C):
                raise ValueError(f"Target shape must be {(N, C)} or (N,) - got {target.shape}")
            base_target = target.astype(np.float32)

        # Apply label smoothing
        self.target = base_target * (1.0 - self.smoothing) + self.smoothing / float(C)

        loss = -np.mean(np.sum(self.target * np.log(self.softmax + eps), axis=-1))
        return float(loss)

    def backward(self) -> np.ndarray:
        self.gradient = (self.softmax - self.target) / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Label Smoothing Cross Entropy"

class DiceLoss(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                eps: float = 1e-9) -> float:
        super().forward(input, target)
        self.eps = eps
        input_flat = input.reshape(input.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        intersection = np.sum(input_flat * target_flat, axis=1)
        dice = (2. * intersection + eps) / (np.sum(input_flat, axis=1) + np.sum(target_flat, axis=1) + eps)
        loss = 1 - np.mean(dice)
        return loss

    def backward(self) -> np.ndarray:
        input_flat = self.input.reshape(self.input.shape[0], -1)
        target_flat = self.target.reshape(self.target.shape[0], -1)
        intersection = np.sum(input_flat * target_flat, axis=1, keepdims=True)
        denom = (np.sum(input_flat, axis=1, keepdims=True) + np.sum(target_flat, axis=1, keepdims=True) + self.eps)
        grad = -2 * (target_flat * denom - 2 * intersection * input_flat) / (denom ** 2)
        self.gradient = grad.reshape(self.input.shape) / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Dice Loss"
