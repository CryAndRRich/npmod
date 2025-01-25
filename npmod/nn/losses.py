import numpy as np

class Loss:
    """
    Base class for loss functions
    """
    def __init__(self):
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

        --------------------------------------------------
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

    def __str__(self):
        pass


class MAE(Loss):
    def forward(self, input, target):
        super().forward(input, target)
        loss = np.mean(np.abs(self.input - self.target))
        return loss

    def backward(self):
        self.gradient = np.sign(self.input - self.target) / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Mean Absolute Error (MAE)"


class MSE(Loss):
    def forward(self, input, target):
        super().forward(input, target)
        loss = np.mean((self.input - self.target) ** 2)
        return loss

    def backward(self):
        self.gradient = 2 * (self.input - self.target) / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Mean Squared Error (MSE)"


class MALE(Loss):
    def forward(self, input, target):
        assert np.any(input > 0) and np.any(target > 0), \
            "Input and target must be greater than 0 for log transformation"
        
        super().forward(input, target)
        loss = np.mean((np.log(self.input) - np.log(self.target)) ** 2)
        return loss

    def backward(self):
        assert np.any(self.input > 0) and np.any(self.target > 0), \
            "Input and target must be greater than 0 for log transformation"
        
        self.gradient = 2 * (np.log(self.input) - np.log(self.target)) / (self.input * self.input.shape[0])
        return self.gradient

    def __str__(self):
        return "Mean Absolute Log Error (MALE)"


class RSquared(Loss):
    def forward(self, input, target):
        super().forward(input, target)
        mean = np.mean(self.input)
        numerator = np.sum((self.target - self.input) ** 2)
        dominator = np.sum((self.target - mean) ** 2)
        if dominator == 0:
            raise ValueError("The denominator is zero, cannot compute R²")
        
        loss = numerator / dominator
        return loss

    def backward(self):
        mean = np.mean(self.input)
        dominator = np.sum((self.target - mean) ** 2)
        if dominator == 0:
            raise ValueError("The denominator is zero, cannot compute self.gradient")
        self.gradient = -2 * (self.target - self.input) / dominator
        return self.gradient

    def __str__(self):
        return "R Squared (R2)"


class MAPE(Loss):
    def forward(self, input, target):
        eps = 1e-9
        super().forward(input, target)
        ratio = np.abs(1 - self.input / (self.target + eps))
        loss = np.mean(ratio)
        return loss

    def backward(self):
        eps = 1e-9
        self.gradient = -1 / (self.target + eps) / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Mean Absolute Percentage Error (MAPE)"


class wMAPE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                weights: float | np.ndarray = None) -> float:

        if weights is None:
            weights = np.ones(input.shape[0])

        assert isinstance(weights, (float, np.ndarray)), \
            "Weights must be a float or numpy array"
        
        if isinstance(weights, np.ndarray) and weights.shape[0] != input.shape[0]:
            raise ValueError("Weights must have the same shape as input")
        
        self.weights = weights
        eps = 1e-9 # Epsilon to avoid divided by zero
        super().forward(input, target)
        numerator = np.sum(np.abs(self.target - self.input) * weights)
        dominator = np.sum(np.abs(self.target) * weights) + eps
        loss = numerator / dominator
        return loss

    def backward(self):
        eps = 1e-9 # Epsilon to avoid divided by zero
        self.gradient = -self.weights / (np.sum(np.abs(self.target) * self.weights) + eps)
        return self.gradient

    def __str__(self):
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
    
    def backward(self):
        ln = self.input - self.target
        self.gradient = np.where(np.abs(ln) < self.beta, ln, np.sign(ln))
        return self.gradient

    def __str__(self):
        return "Smooth L1 Loss (SmoothL1)"


class CE(Loss):
    def forward(self, input, target):
        super().forward(input, target)
        eps = 1e-9 # Epsilon to avoid log(0)
        entropy = -target * np.log(input + eps)
        loss = np.mean(entropy)
        return loss
    
    def backward(self):
        ones_for_targets = np.zeros_like(self.input)
        ones_for_targets[np.arange(len(self.input)), self.target] = 1
        softmax = np.exp(self.input) / np.exp(self.input).sum(axis=-1, keepdims=True)

        self.gradient = (-ones_for_targets + softmax) / self.input.shape[0]
        return self.gradient
    
    def __str__(self):
        return "Cross Entropy Loss (CE)"


class BCE(Loss):
    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray, 
                weights: float | np.ndarray = None) -> float:

        if weights is None:
            weights = np.ones(input.shape[0])

        assert isinstance(weights, (float, np.ndarray)), \
            "Weights must be a float or numpy array"
        
        if isinstance(weights, np.ndarray) and weights.shape[0] != input.shape[0]:
            raise ValueError("Weights must have the same shape as input")
        
        self.weights = weights
        eps = 1e-9 # Epsilon to avoid log(0)
        super().forward(input, target)
        entropy = -self.weights * ((input * np.log(target + eps)) + ((1 - input) * (np.log(1 - target + eps))))
        loss = np.mean(entropy)
        return loss
    
    def backward(self):
        eps = 1e-9 # Epsilon to avoid divided by zero
        self.gradient = self.weights * ((self.input - self.target) / ((self.input * (1 - self.input)) + eps))
        return self.gradient
    
    def __str__(self):
        return "Binary Cross Entropy (BCE)"


class KLDiv(Loss):
    def forward(self, input, target):
        super().forward(input, target)
        eps = 1e-9
        loss = np.mean(target * (np.log(target + eps) - input))
        return loss
    
    def backward(self):
        self.gradient = -self.target / self.target.shape[0]
        return self.gradient
    
    def __str__(self):
        return "Kullback-Leibler Divergence Loss (KLDiv)"
