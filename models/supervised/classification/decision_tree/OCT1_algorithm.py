import numpy as np
from .tree import *
from .utils import entropy, information_gain

np.random.seed(42)

class OCT1DecisionTree(Tree):
    """
    Instead of splitting data using simple attribute conditions, OC1DecisionTree uses oblique hyperplanes defined
    by a weight vector and a threshold to separate the data
    """
    def build_tree(self, 
                   features: np.ndarray, 
                   targets: np.ndarray) -> TreeNode:
        
        # If all targets are the same, return a leaf node with that target
        if len(np.unique(targets)) == 1:
            return TreeNode(results=targets[0])
        
        _, n_features = features.shape
        current_entropy = entropy(targets)
        
        best_gain = 0.0
        best_weights = None
        best_threshold = None
        best_sets = None
        
        # Number of random attempts to find the best hyperplane
        n_attempts = 10
        
        # Try multiple random weight vectors to find the optimal oblique hyperplane
        for _ in range(n_attempts):
            weights = np.random.randn(n_features)
            
            # Compute the projection of the data samples onto the weight vector
            projections = features.dot(weights)
            
            # Choose the threshold as the median of the projected values
            threshold = np.median(projections)
            
            # Split the data based on the hyperplane: samples with projection <= threshold go to the true branch
            true_idx = projections <= threshold
            false_idx = projections > threshold
            
            # If one of the branches is empty, skip this attempt
            if np.sum(true_idx) == 0 or np.sum(false_idx) == 0:
                continue
            
            true_features, false_features = features[true_idx], features[false_idx]
            true_targets, false_targets = targets[true_idx], targets[false_idx]
            
            information_gain_value = information_gain(true_targets, false_targets, current_entropy)
            
            if information_gain_value > best_gain:
                best_gain = information_gain_value
                best_weights = weights
                best_threshold = threshold
                best_sets = (true_features, true_targets, false_features, false_targets)
        
        if best_gain > 0 and best_sets is not None:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=None, 
                            value=(best_weights, best_threshold),
                            true_branch=true_branch, 
                            false_branch=false_branch)
        else:
            # If no valid split is found, create a leaf node with the majority target
            counts = {}
            for target in targets:
                counts[target] = counts.get(target, 0) + 1
            majority_target = max(counts.items(), key=lambda x: x[1])[0]
            return TreeNode(results=majority_target)
    
    def predict_node(self, 
                     tree: TreeNode, 
                     sample: np.ndarray) -> int:
        """
        Traverses the decision tree to predict the target for a single data sample
        
        Parameters:
            tree: The current node in the decision tree
            sample: The feature vector of the sample to predict
        
        Returns:
            int: The predicted target for the sample
        """
        # If it's a leaf node, return its target
        if tree.results is not None:
            return tree.results
        else:
            # For non-leaf nodes, tree.value contains a tuple (weights, threshold)
            weights, threshold = tree.value
            # Compute the dot product of the sample with the weight vector and compare with the threshold
            if np.dot(sample, weights) <= threshold:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
            return self.predict_node(branch, sample)
    
    def __str__(self) -> str:
        return "OCT1 Algorithm"
