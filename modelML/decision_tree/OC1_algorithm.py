import numpy as np
from .tree import *
from .utils import entropy, information_gain

np.random.seed(42)

class OC1DecisionTree(Tree):
    """
    Instead of splitting data using simple attribute conditions, OC1DecisionTree uses oblique hyperplanes defined
    by a weight vector and a threshold to separate the data
    """
    def build_tree(self, 
                   features: np.ndarray, 
                   labels: np.ndarray) -> TreeNode:
        """
        Recursively builds the oblique decision tree using the OC1 algorithm
        
        --------------------------------------------------
        Parameters:
            features: The feature matrix of the training data with shape (n_samples, n_features)
            labels: The array of labels corresponding to the training data, shape (n_samples,)
        
        --------------------------------------------------
        Returns:
            TreeNode: The root node of the constructed decision tree
        """
        # If all labels are the same, return a leaf node with that label
        if len(np.unique(labels)) == 1:
            return TreeNode(results=labels[0])
        
        _, n_features = features.shape
        current_entropy = entropy(labels)
        
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
            true_labels, false_labels = labels[true_idx], labels[false_idx]
            
            information_gain_value = information_gain(true_labels, false_labels, current_entropy)
            
            if information_gain_value > best_gain:
                best_gain = information_gain_value
                best_weights = weights
                best_threshold = threshold
                best_sets = (true_features, true_labels, false_features, false_labels)
        
        if best_gain > 0 and best_sets is not None:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=None, 
                            value=(best_weights, best_threshold),
                            true_branch=true_branch, 
                            false_branch=false_branch)
        else:
            # If no valid split is found, create a leaf node with the majority label
            counts = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            majority_label = max(counts.items(), key=lambda x: x[1])[0]
            return TreeNode(results=majority_label)
    
    def predict_node(self, 
                     tree: TreeNode, 
                     sample: np.ndarray) -> int:
        """
        Traverses the decision tree to predict the label for a single data sample
        
        --------------------------------------------------
        Parameters:
            tree: The current node in the decision tree.
            sample: The feature vector of the sample to predict
        
        --------------------------------------------------
        Returns:
            int: The predicted label for the sample.
        """
        # If it's a leaf node, return its label
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
        return "Decision Trees: OC1 Algorithm"
