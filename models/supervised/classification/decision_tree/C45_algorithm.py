import numpy as np
from .tree import *
from .utils import entropy, information_gain, split_data

class C45DecisionTree(Tree):
    def build_tree(self, 
                   features: np.ndarray, 
                   targets: np.ndarray) -> TreeNode:
        best_gain_ratio = 0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        current_entropy = entropy(targets)

        # Iterate over each feature to find the best split
        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_targets, false_features, false_targets = split_data(features, targets, feature, value)
                gain_ratio_value = information_gain(true_targets, false_targets, current_entropy, get_ratio=True)

                if gain_ratio_value > best_gain_ratio:
                    best_gain_ratio = gain_ratio_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_targets, false_features, false_targets)

        # If a valid split is found, create branches recursively
        if best_gain_ratio > 0:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_criteria[0], 
                            value=best_criteria[1], 
                            true_branch=true_branch, 
                            false_branch=false_branch)

        # If no further split is possible, return a leaf node with the most common label
        return TreeNode(results=np.argmax(np.bincount(targets)))

    def __str__(self) -> str:
        return "C4.5 Algorithm"
