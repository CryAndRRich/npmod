import numpy as np

class TreeNode():
    def __init__(self, 
                 feature: int = None, 
                 value = None, 
                 results: int = None, 
                 true_branch: "TreeNode" = None, 
                 false_branch: "TreeNode" = None, 
                 samples: int = None, 
                 chi_square: float = None) -> None:
        """
        Parameters:
            feature: Index of the feature used for splitting
            value: Threshold value of the feature to split on
            results: Class target if the node is a leaf
            true_branch: Branch for samples satisfying the split condition
            false_branch: Branch for samples not satisfying the split condition
            samples: Number of samples at the node (for C5.0 algorithm)
            chi_square: Chi-square statistic for the split (for CHAID algorithm)
        """
        self.feature = feature
        self.value = value
        self.results = results
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.samples = samples
        self.chi_square = chi_square

class Tree():
    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Builds the decision tree using the provided training data

        Parameters:
            features: Feature matrix of the training data
            targets: Array of targets corresponding to the training data
        """
        self.decision_tree = self.build_tree(features, targets)
    
    def build_tree(self, 
                   features: np.ndarray, 
                   targets: np.ndarray) -> TreeNode:
        """
        Recursively builds the decision tree

        Parameters:
            features: Feature matrix
            targets: Array of targets corresponding to the features

        Returns:
            TreeNode: The root node of the constructed decision tree
        """
        pass

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        num_samples, _ = test_features.shape

        predictions = np.zeros(num_samples)
        for ind, feature in enumerate(test_features):
            predictions[ind] = self.predict_node(self.decision_tree, feature)

        return predictions
    
    def predict_node(self, 
                     tree: TreeNode, 
                     sample: np.ndarray) -> int:
        """
        Traverses the decision tree to make a prediction for a single sample

        Parameters:
            tree: The current node in the decision tree
            sample: Single feature vector to predict

        Returns:
            int: Predicted class target
        """
        if tree.results is not None:
            return tree.results
        else:
            branch = tree.false_branch
            if sample[tree.feature] <= tree.value:
                branch = tree.true_branch
            return self.predict_node(branch, sample)
    
    def __str__(self) -> str:
        """Returns the string representation of the decision tree"""
        pass
