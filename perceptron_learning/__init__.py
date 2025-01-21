import numpy as np
from .perceptron_numpy import PerceptronLearningNumpy
from .perceptron_pytorch import PerceptronLearningPytorch

class PerceptronLearning():
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int, 
                 type: str = "numpy"):
        """
        Initializes the Perceptron Learning by selecting the appropriate implementation type

        Parameters:
        learn_rate: Learning rate for gradient descent optimization
        number_of_epochs: Number of iterations (epochs) for training
        type: Type of implementation ("numpy" or "pytorch")
        """
        if type == "numpy":
            self.inherit = PerceptronLearningNumpy(learn_rate, number_of_epochs)
        elif type == "pytorch":
            self.inherit = PerceptronLearningPytorch(learn_rate, number_of_epochs)
        else: 
            raise ValueError(f"Type must be 'numpy' or 'pytorch'")
    
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        
        self.inherit.fit(features, labels)

    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray) -> None:
        
        self.inherit.predict(test_features, test_labels)
    
    def __str__(self):
        return self.inherit.__str__()