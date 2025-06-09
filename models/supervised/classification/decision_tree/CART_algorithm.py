import numpy as np
from .tree import *
from .utils import gini_impurity, gini_index, split_data

class CARTDecisionTree(Tree):
    def build_tree(self, 
                   features: np.ndarray, 
                   targets: np.ndarray) -> TreeNode:
        
        best_gini_index = 0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        current_uncertainty = gini_impurity(targets)

        # Iterate over each feature to find the best split
        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_targets, false_features, false_targets = split_data(features, targets, feature, value)
                gini_index_value = gini_index(true_targets, false_targets, current_uncertainty)

                if gini_index_value > best_gini_index:
                    best_gini_index = gini_index_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_targets, false_features, false_targets)

        # If a valid split is found, create branches recursively
        if best_gini_index > 0:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)

        # If no further split is possible, return a leaf node with the most common label
        return TreeNode(results=np.argmax(np.bincount(targets)))

    def __str__(self) -> str:
        return "CART Algorithm"
