from .ID3_algorithm import ID3DecisionTree
from .C45_algorithm import C45DecisionTree
from .C50_algorithm import C50DecisionTree
from .CART_algorithm import CARTDecisionTree
from .CHAID_algorithm import CHAIDDecisionTree
from .CITs_algorithm import CITsDecisionTree

class DecisionTree():
    def __init__(self, algorithm="ID3"):
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
        else: 
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported types of algorithm are 'ID3', 'C4.5', 'C5.0/See5', 'CART', 'CHAID' and 'CITs'")
    
    def fit(self, features, labels):
        self.inherit.fit(features, labels)

    def predict(self, test_features, test_labels):
        self.inherit.predict(test_features, test_labels)
    
    def __str__(self):
        return self.inherit.__str__()