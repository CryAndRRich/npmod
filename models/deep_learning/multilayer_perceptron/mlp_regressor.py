from typing import List, Tuple
import numpy as np

np.random.seed(42)

def relu(x: np.ndarray) -> np.ndarray:
    """Applies the ReLU activation function element-wise"""
    return np.maximum(0, x)

def relu_backward(x: np.ndarray) -> np.ndarray:
    """Computes the derivative of the ReLU activation function element-wise"""
    return (x > 0).astype(float)

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes the Mean Squared Error loss between predictions and true targets"""
    return np.mean((pred - target) ** 2)


class MLPRegressor():
    def __init__(self,
                 input_dim: int,
                 hidden_layers: List[int],
                 output_dim: int = 1,
                 batch_size: int = 32,
                 lr: float = 0.01,
                 epochs: int = 10) -> None:
        """
        Initializes the Multilayer Perceptron model for regression

        Parameters:
            input_dim: Number of input features
            hidden_layers: List containing the number of neurons in each hidden layer
            output_dim: Number of output values (1 for scalar regression, >1 for multi-output)
            batch_size: Size of each training batch
            lr: Learning rate for weight updates
            epochs: Number of training epochs
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # Weight initialization (Xavier/He)
        layer_dims = [input_dim] + hidden_layers + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i+1]
            limit = np.sqrt(6 / (fan_in + fan_out))   # Xavier init
            W = np.random.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, batch: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform a forward pass through the network
        
        Parameters:
            batch : Matrix of inputs

        Returns:
            activations: List of activations for each layer
            pre_activations: List of pre-activation values for each layer
        """
        activations = [batch]
        pre_activations = []
        for i in range(len(self.weights) - 1):
            Z = activations[-1] @ self.weights[i] + self.biases[i]
            A = relu(Z)
            pre_activations.append(Z)
            activations.append(A)

        # Output layer (linear activation for regression)
        Z = activations[-1] @ self.weights[-1] + self.biases[-1]
        A = Z
        pre_activations.append(Z)
        activations.append(A)

        return activations, pre_activations
    
    def backward(self, 
                 activations: List[np.ndarray],
                 pre_activations: List[np.ndarray],
                 y_true: np.ndarray) -> None:
        """
        Performs backpropagation and updates weights and biases
        
        Parameters:
            activations : List of activations from each layer
            pre_activations : List of pre-activation values from each layer
            y_true : True targets
        """
        grads_W = []
        grads_b = []

        # output layer error (MSE derivative: dL/dZ = (y_pred - y_true))
        delta = (activations[-1] - y_true)
        for i in reversed(range(len(self.weights))):
            A_prev = activations[i]
            dW = A_prev.T @ delta / A_prev.shape[0]
            db = np.mean(delta, axis=0, keepdims=True)

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:  # backprop to hidden layers
                dA = delta @ self.weights[i].T
                delta = dA * relu_backward(pre_activations[i-1])

        # update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]
    
    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Trains the MLP regression model using the training data

        Parameters:
            features: Input feature matrix for training
            targets: Continuous target values
        """
        n_samples = features.shape[0]
        y_true = targets.reshape(n_samples, self.output_dim)

        for _ in range(self.epochs):
            # shuffle
            idx = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = features[idx], y_true[idx]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                activations, pre_activations = self.forward(X_batch)
                self.backward(activations, pre_activations, y_batch)
    
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test set

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The predicted continuous values
        """
        activations, _ = self.forward(test_features)
        return activations[-1]
    
    def __str__(self) -> str:
        return "Multilayer Perceptron Regressor"
