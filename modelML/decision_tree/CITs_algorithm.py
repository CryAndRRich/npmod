import numpy as np
from .tree import *
from .utils import chi_square, chi_square_p_value, split_data

class CITsDecisionTree(Tree):
    def __init__(self, 
                 min_samples_split: int = 2, 
                 p_value_threshold: float = 0.05):
        """
        Initializes the CITs decision tree

        --------------------------------------------------
        Parameters:
            min_samples_split: Minimum number of samples required to split a node
            p_value_threshold: Threshold for the p-value to determine significant splits
        """
        self.min_samples_split = min_samples_split
        self.p_value_threshold = p_value_threshold
        self.tree = None

    def build_tree(self, 
                   features: np.ndarray, 
                   labels: np.ndarray) -> TreeNode:
        
        # Stop splitting if the number of samples is less than the minimum required
        if len(labels) < self.min_samples_split:
            return TreeNode(results=np.argmax(np.bincount(labels)))
        
        best_p_value = 1
        best_criteria = None
        best_sets = None
        _, n = features.shape

        # Iterate over each feature to find the best split
        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_labels, false_features, false_labels = split_data(features, labels, feature, value)

                if len(true_labels) == 0 or len(false_labels) == 0:
                    continue
                
                chi_square_value = chi_square(true_labels, false_labels, labels)
                p_value = chi_square_p_value(chi_square_value, df=len(np.unique(labels)))

                if p_value < best_p_value:
                    best_p_value = p_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_labels, false_features, false_labels)

        # If a valid split is found, create branches recursively
        if best_p_value < self.p_value_threshold:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)

        # If no further split is possible, return a leaf node with the most common label
        return TreeNode(results=np.argmax(np.bincount(labels)))

    def __str__(self):
        return "Decision Trees: CITs Algorithm"
