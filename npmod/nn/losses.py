import numpy as np

class Loss():
    """
    Base class for loss functions
    """
    def __init__(self) -> None:
        self.input = None
        self.target = None

    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> float:
        """
        Forward pass of the loss function

        Parameters:
            input: Predictions of Layer/Model
            target: Targets to be evaluated against

        Returns:
            float: A measure of prediction accuracy
        """
        self.input = input
        self.target = target
        pass

    def backward(self) -> np.ndarray:
        """
        Backward pass of the loss function

        Returns:
            np.ndarray: Compute gradient of the loss function with respect to input
        """
        pass

    def __call__(self, input, target) -> float:
        return self.forward(input, target)

    def __str__(self) -> str:
        pass

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
            raise ValueError("The denominator is zero, cannot compute R²")
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

class CE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> np.ndarray:
        super().forward(input, target)
        eps = 1e-9
        self.input_stable = input - np.max(input, axis=-1, keepdims=True)
        exp_input = np.exp(self.input_stable)
        self.softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)

        # convert target to one-hot if necessary
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
