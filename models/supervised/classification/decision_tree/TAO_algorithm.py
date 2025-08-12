from typing import Any, Tuple
import numpy as np
from .tree import *
from .utils import entropy, information_gain, split_data

class TAODecisionTree(Tree):
    def __init__(self, 
                 max_iterations: int = 10, 
                 threshold: float = 1e-4) -> str:
        """
        Initializes the TAO decision tree

        Parameters:
            max_iterations: Maximum number of optimization iterations
            threshold: Threshold for improvement to continue optimization
        """
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.tree = None

    def build_tree(self, 
                   features: np.ndarray, 
                   targets: np.ndarray) -> TreeNode:
        """
        Builds the decision tree using an initial split and optimizes using TAO

        Parameters:
            features: Feature matrix
            targets: Array of targets corresponding to the features

        Returns:
            TreeNode: The root node of the constructed and optimized decision tree
        """
        initial_tree = self.build_initial_tree(features, targets)
        return self.optimize_tree(initial_tree, features, targets)

    def build_initial_tree(self, 
                           features: np.ndarray, 
                           targets: np.ndarray) -> TreeNode:
        """
        Builds an initial decision tree using entropy-based splitting

        Returns:
            TreeNode: The root node of the initial decision tree
        """
        best_gain = 0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        current_entropy = entropy(targets)

        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_targets, false_features, false_targets = split_data(features, targets, feature, value)
                gain = information_gain(true_targets, false_targets, current_entropy)

                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_targets, false_features, false_targets)

        if best_gain > 0:
            true_branch = self.build_initial_tree(best_sets[0], best_sets[1])
            false_branch = self.build_initial_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)
        
        return TreeNode(results=np.argmax(np.bincount(targets)))

    def optimize_tree(self, 
                      tree: TreeNode, 
                      features: np.ndarray, 
                      targets: np.ndarray) -> TreeNode:
        """
        Optimizes the decision tree using the TAO algorithm

        Parameters:
            tree: The initial decision tree
            features: Feature matrix
            targets: Array of targets corresponding to the features

        Returns:
            TreeNode: The optimized decision tree
        """
        for _ in range(self.max_iterations):
            improved_overall = False
            for node in self.iterate_nodes(tree):
                if node.results is None:
                    best_gain = 0
                    best_value = node.value
                    best_feature = node.feature

                    node_features, node_targets = self.get_node_data(features, targets, node)
                    current_entropy = entropy(node_targets)
                    improved_node = False

                    for value in set(node_features[:, node.feature]):
                        _, true_targets, _, false_targets = split_data(node_features, node_targets, node.feature, value)
                        gain = information_gain(true_targets, false_targets, current_entropy)

                        if gain > best_gain:
                            best_gain = gain
                            best_value = value
                            best_feature = node.feature
                            improved_node = True

                    if improved_node and best_gain > self.threshold:
                        node.feature = best_feature
                        node.value = best_value
                        improved_overall = True

            if not improved_overall:
                break
        return tree

    def iterate_nodes(self, tree: TreeNode) -> Any:
        """
        Generator function to iterate over all nodes in the tree

        Parameters:
            tree: The root node of the decision tree

        Yields:
            TreeNode: Each node in the decision tree
        """
        if tree is not None:
            yield tree
            yield from self.iterate_nodes(tree.true_branch)
            yield from self.iterate_nodes(tree.false_branch)

    def get_node_data(self, 
                      features: np.ndarray, 
                      targets: np.ndarray, 
                      node: TreeNode) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves feature and target subsets for a given tree node

        Parameters:
            features: Feature matrix
            targets: Array of targets corresponding to the features
            node: TreeNode for which data is retrieved

        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature subset and corresponding targets for the node
        """
        mask = features[:, node.feature] <= node.value
        return features[mask], targets[mask]

    def __str__(self) -> str:
        return "TAO Algorithm"