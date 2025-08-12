import numpy as np

class PerceptronLearning():
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int) -> None:
        """
        Initializes the Perceptron Learning model using the Perceptron Learning Algorithm

        Parameters:
            learn_rate: The learning rate for the model update
            number_of_epochs: The number of training iterations
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs

    def heaviside_step(self, x):
        return np.where(x >= 0, 1, 0)
    
    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Trains the Perceptron model using the training data

        Parameters:
            features: Input feature matrix for training
            targets: True targets corresponding to the input features
        """
        _, n_features = features.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.number_of_epochs):
            errors = 0
            for xi, target in zip(features, targets):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = self.heaviside_step(linear_output)
                error = target - prediction
                if error != 0:
                    self.weights += self.learn_rate * error * xi
                    self.bias += self.learn_rate * error
                    errors += 1
            if errors == 0:  # Early stopping
                break
            
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        # Make predictions by applying the Heaviside step function
        linear_output = np.dot(test_features, self.weights) + self.bias
        return self.heaviside_step(linear_output)
    
    def __str__(self) -> str:
        return "Perceptron Learning Algorithm"
