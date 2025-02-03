import numpy as np
import random
from ..decision_tree import *
from ..base_model import ModelML

random.seed(42)

class RandomForest(ModelML):
    def __init__(self, n_tree: int = 10) -> None:
        """
        Initialize a Random Forest with multiple decision trees
        
        --------------------------------------------------
        Parameters:
            n_tree: Number of trees in the forest
        """
        self.n_tree = n_tree
        self.trees = []
        self.tree_names = []

        self.algorithms = ["ID3", "C4.5", "C5.0", "CART", "CHAID", "CITs", "OC1", "QUEST", "TAO"]
        # Initialize a counter for each algorithm
        algo_counter = {algo: 0 for algo in self.algorithms}
        # Determine the maximum threshold for each algorithm:
        # If n_tree > 1, each algorithm is allowed to appear at most n_tree // 2 times
        # If n_tree == 1, there is only one tree, so no restriction is needed
        threshold = n_tree if n_tree == 1 else n_tree // 2
        
        for _ in range(self.n_tree):
            # Get the list of algorithms that have not yet reached the maximum threshold
            available_algos = [algo for algo in self.algorithms if algo_counter[algo] < threshold]
            
            chosen_algo = random.choice(available_algos)
            algo_counter[chosen_algo] += 1
            
            tree = DecisionTree(algorithm=chosen_algo)
            self.trees.append(tree)
            self.tree_names.append(str(tree))
    
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Train the Random Forest by creating multiple decision trees
        
        --------------------------------------------------
        Parameters:
            features: Feature matrix of the training data
            labels: Array of labels corresponding to the training data
        """
        n_samples, _ = features.shape
        
        for ind in range(self.n_tree):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_features = features[indices]
            sample_labels = labels[indices]
            
            # Build a decision tree
            tree = self.trees[ind]
            tree.fit(sample_features, sample_labels)
    
    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray,
                get_accuracy: bool = True) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        --------------------------------------------------
        Parameters:
            test_features: The input features for testing
            test_labels: The true target labels corresponding to the test features
            get_accuracy: If True, calculates and prints the accuracy of predictions

        --------------------------------------------------
        Returns:
            predictions: The prediction labels
        """
        num_samples, _ = test_features.shape
        tree_predictions = [tree.predict(test_features, test_labels, get_accuracy=False) for tree in self.trees]

        final_predictions = np.zeros(num_samples)
        for i in range(num_samples):
            predictions = {}
            for j in range(self.n_tree):
                pred = tree_predictions[j][i]
                if pred not in predictions:
                    predictions[pred] = 0
                predictions[pred] += 1
            
            final_predictions[i] = sorted([(val, key) for key, val in predictions.items()], reverse=True)[0][1]
        
        if get_accuracy:
            accuracy, f1 = self.evaluate(final_predictions, test_labels)
            print("Accuracy: {:.5f} F1-score: {:.5f}".format(accuracy, f1))

        return final_predictions
    
    def __str__(self) -> str:
        self.forest = """Random Forest:\n"""
        for tree in self.tree_names:
            self.forest += "- " + tree + "\n"

        return self.forest
