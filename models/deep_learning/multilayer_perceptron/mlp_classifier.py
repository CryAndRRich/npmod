from typing import List, Tuple
import numpy as np

np.random.seed(42)

def relu(x: np.ndarray) -> np.ndarray:
    """Applies the ReLU activation function element-wise"""
    return np.maximum(0, x)

def relu_backward(x: np.ndarray) -> np.ndarray:
    """Computes the derivative of the ReLU activation function element-wise"""
    return (x > 0).astype(float)

def softmax(z: np.ndarray) -> np.ndarray:
    """Applies the softmax function to each row of the input array"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(pred: np.ndarray, 
                  target: np.ndarray) -> float:
    """Computes the cross-entropy loss between predicted probabilities and true targets"""
    eps = 1e-12
    return -np.sum(target * np.log(pred + eps)) / pred.shape[0]

class MLPClassifier():
    def __init__(self,
                 input_dim: int,
                 hidden_layers: List[int],
                 output_dim: int,
                 batch_size: int = 32,
                 lr: float = 0.01,
                 epochs: int = 10) -> None:
        """
        Initializes the Multilayer Perceptron model

        Parameters:
            input_dim: Number of input features
            hidden_layers: List containing the number of neurons in each hidden layer
            output_dim: Number of output classes
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
        """
        activations = [batch]
        pre_activations = []
        for i in range(len(self.weights) - 1):
            Z = activations[-1] @ self.weights[i] + self.biases[i]
            A = relu(Z)
            pre_activations.append(Z)
            activations.append(A)

        # output layer
        Z = activations[-1] @ self.weights[-1] + self.biases[-1]
        A = softmax(Z)
        pre_activations.append(Z)
        activations.append(A)

        return activations, pre_activations
    
    def backward(self, 
                 activations: np.ndarray,
                 pre_activations: np.ndarray,
                 y_true: np.ndarray) -> None:
        """
        Calculate derivative of sigmoid activation based on sigmoid output
        
        Parameters:
            activations : List of activations from each layer
            pre_activations : List of pre-activation values from each layer
            y_true : True labels in one-hot encoded format
        """
        # Update the weights of the network through back-propagation
        grads_W = []
        grads_b = []

        # output layer error
        delta = activations[-1] - y_true
        for i in reversed(range(len(self.weights))):
            A_prev = activations[i]
            dW = A_prev.T @ delta / A_prev.shape[0]
            db = np.mean(delta, axis=0, keepdims=True)

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:  # backprop to hidden
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
        Trains the MLP model using the training data

        Parameters:
            features: Input feature matrix for training
            targets: True targets corresponding to the input features
        """
        n_samples = features.shape[0]
        y_onehot = np.eye(self.output_dim)[targets.astype(int)]

        for _ in range(self.epochs):
            # shuffle
            idx = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = features[idx], y_onehot[idx]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                activations, pre_activations = self.forward(X_batch)
                self.backward(activations, pre_activations, y_batch)
    
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        activations, _ = self.forward(test_features)
        probs = activations[-1]
        return np.argmax(probs, axis=1)
    
    def __str__(self) -> str:
        return "Multilayer Perceptron Classifier"