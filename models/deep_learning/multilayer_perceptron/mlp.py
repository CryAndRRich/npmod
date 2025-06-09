from typing import List
import numpy as np

np.random.seed(42)

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function for input values

    Parameters:
        x: The input feature values 

    Returns:
        np.ndarray: The sigmoid output for the input values
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x: np.ndarray) -> np.ndarray:
    """
    Calculate derivative of sigmoid activation based on sigmoid output

    Parameters
        x: Output values processed by a sigmoid function
    
    Returns
        np.ndarray: Derivative of sigmoid, based on value of sigmoid
    """
    return x * (1 - x)

def softmax_function(z: np.ndarray) -> np.ndarray:
    """
    Computes the softmax function for the given input scores

    Parameters:
        z: The input score matrix (before softmax)

    Returns:
        np.ndarray: The output probability distribution after applying softmax
    """
    exp_z = np.exp(z - np.max(z))  # Numerical stability by subtracting max(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cost_function(predictions: np.ndarray,
                  targets: np.ndarray) -> float:
    """
    Computes the loss between predictions and true targets

    Parameters:
        predictions: Predicted targets
        targets: True targets 

    Returns:
        float: The loss between predictions and targets
    """
    return ((-np.log(predictions)) * targets).sum(axis=1).mean()


class MLP():
    def __init__(self,
                 batch_size: int,
                 learn_rate: float, 
                 number_of_epochs: int,
                 n_layers: int,
                 n_neurons: List[int]) -> None:
        """
        Initializes the Multilayer Perceptron model

        Parameters:
            batch_size: Size of a training mini-batch
            learn_rate: The learning rate for the model update
            number_of_epochs: The number of training iterations
            n_layers: Number of hidden layers
            n_neurons: Number of neurons in each hidden layer
        """
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.n_layers = n_layers

        assert len(n_neurons) == n_layers, \
            "Number of neurons in each hidden layer must equal to Number of hidden layers"
        self.n_neurons = n_neurons

        self.hidden_layers = list()
        self.weights = list()

    def _hidden_layers(self) -> None:
        """Initialize and allocate arrays for the hidden layer activations"""
        self.hidden_layers = [np.empty((self.batch_size, layer_size)) for layer_size in self.layer_sizes]
    
    def _weights(self) -> None:
        """Initialize the weights of the network given the sizes of the layers"""
        self.weights = list()
        for i in range(self.layer_sizes.shape[0] - 1):
            self.weights.append(np.random.uniform(-1, 1, size=[self.layer_sizes[i], self.layer_sizes[i+1]]))

    def _categorical(self, probs: np.ndarray) -> np.ndarray:  
        """Transform probabilities into categorical predictions row-wise, by simply taking the max probability"""
        categorical = np.zeros((probs.shape[0], self.targets.shape[1]))
        categorical[np.arange(probs.shape[0]), probs.argmax(axis=1)] = 1
        return categorical
    
    def forward(self, batch: np.ndarray) -> None:
        """
        Perform a forward pass of 'batch' samples (n_samples, n_features)
        
        Parameters:
            batch : Matrix of inputs
        """
        h_l = batch
        self.hidden_layers[0] = h_l
        for i, weights in enumerate(self.weights):
            h_l = sigmoid_function(h_l.dot(weights))
            self.hidden_layers[i + 1] = h_l

        # Forward pass output of the MLP
        self.output = softmax_function(self.hidden_layers[-1])
    
    def backward(self, batch: np.ndarray) -> None:
        """
        Calculate derivative of sigmoid activation based on sigmoid output
        
        Parameters:
            batch : True targets for the samples in the batch
        """
        # Update the weights of the network through back-propagation
        delta_t = (self.output - batch) * sigmoid_backward(self.hidden_layers[-1])
        for i in range(1, len(self.weights) + 1):
            self.weights[-i] -= self.learn_rate * (self.hidden_layers[-i - 1].T.dot(delta_t)) / self.batch_size
            delta_t = sigmoid_backward(self.hidden_layers[-i - 1]) * (delta_t.dot(self.weights[-i].T))
    
    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Trains the MLP model using the training data

        Parameters:
            features: Input feature matrix for training
            targets: True targets corresponding to the input features
        """
        features = features.astype(np.float32)
        self.features = np.concatenate((features, np.ones((features.shape[0], 1), dtype=np.float32)), axis=1)
        self.targets = np.squeeze(np.eye(10)[targets.astype(np.int32).reshape(-1)])

        self.n_samples = self.features.shape[0]
        self.layer_sizes = np.array([self.features.shape[1]] + self.n_neurons + [self.targets.shape[1]])
        self._weights()

        for _ in range(self.number_of_epochs):
            self._hidden_layers()
            shuffle = np.random.permutation(self.n_samples)

            features_batches = np.array_split(self.features[shuffle], self.n_samples // self.batch_size)
            targets_batches = np.array_split(self.targets[shuffle], self.n_samples // self.batch_size)

            for features_batch, targets_batch in zip(features_batches, targets_batches):
                self.forward(features_batch)  
                self.backward(targets_batch)
    
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        test_features = test_features.astype(np.float32)
        self.test_features = np.concatenate((test_features, np.ones((test_features.shape[0], 1), dtype=np.float32)), axis=1)

        self.forward(self.test_features)
        predictions = self._categorical(self.output)

        return predictions
    
    def __str__(self) -> str:
        return "Multilayer Perceptron"