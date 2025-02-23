import numpy as np
from .tree import *
from .utils import chi_square, split_data

class CITsDecisionTree(Tree):
    def __init__(self, 
                 min_samples_split: int = 2, 
                 p_value_threshold: float = 0.05,
                 num_permutations: int = 100):
        """
        Initializes the CITs decision tree

        --------------------------------------------------
        Parameters:
            min_samples_split: Minimum number of samples required to split a node
            p_value_threshold: Threshold for the p-value to determine significant splits
            num_permutations: Number of permutations to perform for the permutation test
        """
        self.min_samples_split = min_samples_split
        self.p_value_threshold = p_value_threshold
        self.num_permutations = num_permutations
        self.tree = None

    def build_tree(self, 
                   features: np.ndarray, 
                   labels: np.ndarray) -> TreeNode:
        """
        Recursively builds the CITs decision tree with permutation tests and 
        Bonferroni correction for multiple testing

        --------------------------------------------------
        Parameters:
            features: Feature matrix of the training data
            labels: Array of labels corresponding to the training data

        --------------------------------------------------
        Returns:
            TreeNode: The root node of the constructed decision tree
        """
        # Stop splitting if the number of samples is less than the minimum required
        if len(labels) < self.min_samples_split:
            return TreeNode(results=np.argmax(np.bincount(labels)))
        
        candidate_p_values = []  # To store permutation p-values for each candidate split
        candidate_splits = []    # To store candidate splits (feature, value, split data)

        _, n = features.shape

        # Iterate over each feature to evaluate all possible candidate splits
        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                # Split the data based on the current candidate split
                true_features, true_labels, false_features, false_labels = split_data(features, labels, feature, value)
                
                # Skip candidate splits that result in an empty branch
                if len(true_labels) == 0 or len(false_labels) == 0:
                    continue

                # Compute the chi-square statistic for the candidate split
                chi_square_value = chi_square(true_labels, false_labels, labels)
                
                # Perform a permutation test to obtain a robust p-value
                # We build a null distribution by randomly permuting the labels
                permuted_count = 0
                for _ in range(self.num_permutations):
                    # Permute the labels.
                    permuted_labels = np.random.permutation(labels)
                    # Apply the same split to the permuted labels.
                    _, true_labels_perm, _, false_labels_perm = \
                        split_data(features, permuted_labels, feature, value)
                    
                    # Skip this permutation if one branch is empty
                    if len(true_labels_perm) == 0 or len(false_labels_perm) == 0:
                        continue
                    
                    # Compute the chi-square statistic for the permuted split
                    chi_square_perm = chi_square(true_labels_perm, false_labels_perm, permuted_labels)
                    if chi_square_perm >= chi_square_value:
                        permuted_count += 1
                
                # The permutation p-value is the proportion of permutations with
                # a chi-square statistic at least as extreme as the observed one
                permutation_p_value = permuted_count / self.num_permutations

                # Store candidate p-value and split information
                candidate_p_values.append(permutation_p_value)
                candidate_splits.append((feature, value, (true_features, true_labels, false_features, false_labels)))

        # If no candidate splits were found, return a leaf node
        if len(candidate_p_values) == 0:
            return TreeNode(results=np.argmax(np.bincount(labels)))
        
        # Apply Bonferroni correction for multiple testing
        # Multiply each candidate p-value by the number of tests and clip at 1.0
        num_tests = len(candidate_p_values)
        corrected_p_values = [min(p * num_tests, 1.0) for p in candidate_p_values]

        # Select the candidate split with the lowest corrected p-value
        best_index = np.argmin(corrected_p_values)
        best_corrected_p_value = corrected_p_values[best_index]
        best_feature, best_value, best_sets = candidate_splits[best_index]

        # Accept the split only if the corrected p-value is below the threshold
        if best_corrected_p_value < self.p_value_threshold:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_feature, 
                            value=best_value, 
                            true_branch=true_branch, 
                            false_branch=false_branch)

        return TreeNode(results=np.argmax(np.bincount(labels)))

    def __str__(self):
        return "Decision Trees: CITs Algorithm"
