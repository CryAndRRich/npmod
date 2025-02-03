import numpy as np
from .tree import *
from .utils import chi_square, chi_square_p_value, split_data

class QUESTDecisionTree(Tree):
    def build_tree(self, 
                   features: np.ndarray, 
                   labels: np.ndarray) -> TreeNode:
        """
        Builds the QUEST decision tree using chi-square tests to determine the best split

        --------------------------------------------------
        Parameters:
            features: Feature matrix
            labels: Array of labels corresponding to the features

        --------------------------------------------------
        Returns:
            TreeNode: The root node of the constructed decision tree
        """

        best_p_value = 1.0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        # Iterate over each feature to find the best split
        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_labels, false_features, false_labels = split_data(features, labels, feature, value)

                # Compute chi-square statistic
                chi_square_stat = chi_square(true_labels, false_labels, labels)

                df = len(set(labels)) - 1 # Degrees of freedom = number of classes - 1
                p_value = chi_square_p_value(chi_square_stat, df) # Compute p-value

                # Choose the best split based on the lowest p-value
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_labels, false_features, false_labels)

        # If a valid split is found, create branches recursively
        if best_criteria and best_p_value < 0.05:  # Using a significance threshold of 0.05
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_criteria[0], 
                            value=best_criteria[1], 
                            true_branch=true_branch, 
                            false_branch=false_branch)

        # If no further split is possible, return a leaf node with the most common label
        return TreeNode(results=np.argmax(np.bincount(labels)))

    def __str__(self):
        return "Decision Trees: QUEST Algorithm"
