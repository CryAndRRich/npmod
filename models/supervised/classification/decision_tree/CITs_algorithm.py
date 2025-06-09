import numpy as np
from .tree import *
from .utils import chi_square, split_data

class CITsDecisionTree(Tree):
    def __init__(self, 
                 min_samples_split: int = 2, 
                 p_value_threshold: float = 0.05,
                 num_permutations: int = 100) -> None:
        """
        Initializes the CITs decision tree

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
                   targets: np.ndarray) -> TreeNode:
        """
        Recursively builds the CITs decision tree with permutation tests and 
        Bonferroni correction for multiple testing

        Parameters:
            features: Feature matrix of the training data
            targets: Array of targets corresponding to the training data

        Returns:
            TreeNode: The root node of the constructed decision tree
        """
        # Stop splitting if the number of samples is less than the minimum required
        if len(targets) < self.min_samples_split:
            return TreeNode(results=np.argmax(np.bincount(targets)))
        
        candidate_p_values = []  # To store permutation p-values for each candidate split
        candidate_splits = []    # To store candidate splits (feature, value, split data)

        _, n = features.shape

        # Iterate over each feature to evaluate all possible candidate splits
        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                # Split the data based on the current candidate split
                true_features, true_targets, false_features, false_targets = split_data(features, targets, feature, value)
                
                # Skip candidate splits that result in an empty branch
                if len(true_targets) == 0 or len(false_targets) == 0:
                    continue

                # Compute the chi-square statistic for the candidate split
                chi_square_value = chi_square(true_targets, false_targets, targets)
                
                # Perform a permutation test to obtain a robust p-value
                # We build a null distribution by randomly permuting the targets
                permuted_count = 0
                for _ in range(self.num_permutations):
                    # Permute the targets.
                    permuted_targets = np.random.permutation(targets)
                    # Apply the same split to the permuted targets.
                    _, true_targets_perm, _, false_targets_perm = \
                        split_data(features, permuted_targets, feature, value)
                    
                    # Skip this permutation if one branch is empty
                    if len(true_targets_perm) == 0 or len(false_targets_perm) == 0:
                        continue
                    
                    # Compute the chi-square statistic for the permuted split
                    chi_square_perm = chi_square(true_targets_perm, false_targets_perm, permuted_targets)
                    if chi_square_perm >= chi_square_value:
                        permuted_count += 1
                
                # The permutation p-value is the proportion of permutations with
                # a chi-square statistic at least as extreme as the observed one
                permutation_p_value = permuted_count / self.num_permutations

                # Store candidate p-value and split information
                candidate_p_values.append(permutation_p_value)
                candidate_splits.append((feature, value, (true_features, true_targets, false_features, false_targets)))

        # If no candidate splits were found, return a leaf node
        if len(candidate_p_values) == 0:
            return TreeNode(results=np.argmax(np.bincount(targets)))
        
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

        return TreeNode(results=np.argmax(np.bincount(targets)))

    def __str__(self) -> str:
        return "CITs Algorithm"
