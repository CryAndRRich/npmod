import numpy as np
from .tree import *
from .utils import entropy, information_gain, split_data

class C50DecisionTree(Tree):
    def __init__(self, 
                 n_estimators: int = 10, 
                 min_samples: int = 1) -> None:
        """
        Initializes the C5.0 decision tree with boosting

        Parameters:
            n_estimators: Number of boosting iterations
            min_samples: Minimum number of samples required to split a node
        """
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self.trees = []
        self.tree_weights = []

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fits the C5.0 model to the training data using boosting
        """
        n_samples = len(targets)
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            tree = self.build_tree(features, targets, weights)
            predictions = np.array([self.predict_node(tree, sample) for sample in features])
            error = np.sum(weights * (predictions != targets)) / np.sum(weights)
            
            if error >= 0.5:
                continue
            
            tree_weight = 0.5 * np.log((1 - error) / max(error, 1e-10))
            self.trees.append(tree)
            self.tree_weights.append(tree_weight)
            
            weights *= np.exp(-tree_weight * targets * (2 * predictions - 1))
            weights /= np.sum(weights)

    def build_tree(self, 
                   features: np.ndarray, 
                   targets: np.ndarray, 
                   weights: np.ndarray) -> TreeNode:
        """
        Builds a single decision tree using weighted data

        Parameters:
            features: Feature matrix of the training data
            targets: Array of targets corresponding to the training data
            weights: Weights for each sample in the training data

        Returns:
            TreeNode: Root node of the constructed decision tree
        """
        best_gain_ratio = 0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        current_entropy = entropy(targets, weights)

        # Iterate over each feature to find the best split
        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_targets, true_weights, false_features, false_targets, false_weights = split_data(features, targets, feature, value, weights)
                gain_ratio_value = information_gain(true_targets, false_targets, current_entropy, true_weights, false_weights, get_ratio=True)

                if gain_ratio_value > best_gain_ratio:
                    best_gain_ratio = gain_ratio_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_targets, true_weights, false_features, false_targets, false_weights)

        # If a valid split is found, create branches recursively
        if best_gain_ratio > 0:
            true_branch = self.build_tree(best_sets[0], best_sets[1], best_sets[2])
            false_branch = self.build_tree(best_sets[3], best_sets[4], best_sets[5])
            return TreeNode(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch, samples=len(targets))

        # If no further split is possible, return a leaf node with the most common label
        return TreeNode(results=np.argmax(np.bincount(targets, weights=weights)), samples=len(targets))

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        
        num_samples, _ = test_features.shape

        predictions = np.zeros(num_samples)
        for tree, tree_weight in zip(self.trees, self.tree_weights):
            predictions += tree_weight * np.array([self.predict_node(tree, sample) for sample in test_features])

        final_predictions = np.sign(predictions)

        return final_predictions
    
    def __str__(self) -> str:
        return "C5.0/See5 Algorithm"
