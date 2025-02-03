import random
import numpy as np
from .ID3_algorithm import ID3DecisionTree
from .C45_algorithm import C45DecisionTree
from .C50_algorithm import C50DecisionTree
from .CART_algorithm import CARTDecisionTree
from .CHAID_algorithm import CHAIDDecisionTree
from .CITs_algorithm import CITsDecisionTree
from .OC1_algorithm import OC1DecisionTree
from .QUEST_algorithm import QUESTDecisionTree
from .TAO_algorithm import TAODecisionTree

random.seed(42)

class DecisionTree():
    """
    A unified interface for decision tree algorithms. This class acts as a wrapper
    for different decision tree implementations, allowing users to specify the algorithm type
    """

    def __init__(self, algorithm: str = None):
        """
        Initializes the DecisionTree class and selects the appropriate algorithm

        --------------------------------------------------
        Parameters:
            algorithm: The type of decision tree algorithm to use
            Supported values are: 'ID3', 'C4.5', 'C5.0/See5', 'CART', 'CHAID', 'CITs', 'OC1', 'QUEST' and 'TAO'
        """

        self.algorithms = ["ID3", "C4.5", "C5.0", "CART", "CHAID", "CITs", "OC1", "QUEST", "TAO"]
        if algorithm is None:
            tree_type = random.randint(0, len(self.algorithms) - 1)
            algorithm=self.algorithms[tree_type]

        if algorithm == "ID3":
            self.inherit = ID3DecisionTree()
        elif algorithm == "C4.5":
            self.inherit = C45DecisionTree()
        elif algorithm == "C5.0" or algorithm == "See5" or algorithm == "C5.0/See5":
            self.inherit = C50DecisionTree()
        elif algorithm == "CART":
            self.inherit = CARTDecisionTree()
        elif algorithm == "CHAID":
            self.inherit = CHAIDDecisionTree()
        elif algorithm == "CITs":
            self.inherit = CITsDecisionTree()
        elif algorithm == "OC1":
            self.inherit = OC1DecisionTree()
        elif algorithm == "QUEST":
            self.inherit = QUESTDecisionTree()
        elif algorithm == "TAO":
            self.inherit = TAODecisionTree()
        else:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Supported types of algorithm are "
                f"'ID3', 'C4.5', 'C5.0/See5', 'CART', 'CHAID', 'CITs', 'OC1', 'QUEST' and 'TAO'"
            )
    
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        
        self.inherit.fit(features, labels)

    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray,
                get_accuracy: bool = True) -> np.ndarray:
        
        predictions = self.inherit.predict(test_features, test_labels, get_accuracy)
        return predictions
    
    def __str__(self) -> str:
        return self.inherit.__str__()
