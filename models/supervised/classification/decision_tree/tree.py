import numpy as np
from ....base import Model

class TreeNode():
    def __init__(self, 
                 feature: int = None, 
                 value = None, 
                 results: int = None, 
                 true_branch = None, 
                 false_branch = None, 
                 samples: int = None, 
                 chi_square: float = None) -> None:
        """
        Parameters:
            feature: Index of the feature used for splitting
            value: Threshold value of the feature to split on
            results:  Class label if the node is a leaf
            true_branch: (TreeNode) Branch for samples satisfying the split condition
            false_branch: (TreeNode) Branch for samples not satisfying the split condition
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

class Tree(Model):
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Builds the decision tree using the provided training data

        Parameters:
            features: Feature matrix of the training data
            labels: Array of labels corresponding to the training data
        """
        self.decision_tree = self.build_tree(features, labels)
    
    def build_tree(self, 
                   features: np.ndarray, 
                   labels: np.ndarray) -> TreeNode:
        """
        Recursively builds the decision tree

        Parameters:
            features: Feature matrix
            labels: Array of labels corresponding to the features

        Returns:
            TreeNode: The root node of the constructed decision tree
        """
        pass

    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray,
                get_accuracy: bool = True) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing
            test_labels: The true target labels corresponding to the test features
            get_accuracy: If True, calculates and prints the accuracy of predictions

        Returns:
            predictions: The prediction labels
        """
        num_samples, _ = test_features.shape

        predictions = np.zeros(num_samples)
        for ind, feature in enumerate(test_features):
            predictions[ind] = self.predict_node(self.decision_tree, feature)

        if get_accuracy:
            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Accuracy: {:.5f} F1-score: {:.5f}".format(accuracy, f1))

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
            int: Predicted class label
        """
        if tree.results is not None:
            return tree.results
        else:
            branch = tree.false_branch
            if sample[tree.feature] <= tree.value:
                branch = tree.true_branch
            return self.predict_node(branch, sample)
    
    def __str__(self) -> str:
        """
        Returns the string representation of the decision tree
        """
        pass
