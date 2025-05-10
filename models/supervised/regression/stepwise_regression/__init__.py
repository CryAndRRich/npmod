import numpy as np
from .forward_selection import StepwiseForward
from .backward_elimination import StepwiseBackward

class StepwiseRegression():
    def __init__(self,
                 method: str,
                 learn_rate: float,
                 number_of_epochs: int,
                 criterion: str = "mse",
                 threshold: float = 1e-4,
                 verbose: bool = False) -> None:
        """
        Initialize a backward elimination instance

        Parameters:
            method: "forward" or "backward"
            learn_rate: Learning rate for gradient descent in each submodel
            number_of_epochs: Number of training epochs per submodel
            criterion: One of "mse", "r2", "aic" to guide elimination
            threshold: Minimum improvement in the criterion to remove or add a feature
            verbose: If True, prints trial results for each candidate removal
        """
        if method == "forward":
            self.inherit = StepwiseForward(learn_rate, number_of_epochs, criterion, threshold, verbose)
        elif method == "backward":
            self.inherit = StepwiseBackward(learn_rate, number_of_epochs, criterion, threshold, verbose)
        else: 
            raise ValueError(f"Method must be 'forward' or 'backward'")
    
    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        
        self.inherit.fit(features, targets)
    
    def predict(self, 
                test_features: np.ndarray, 
                test_targets: np.ndarray) -> np.ndarray:
        
        self.inherit.predict(test_features, test_targets)

    def __str__(self) -> str:
        return self.inherit.__str__()