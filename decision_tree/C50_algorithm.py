import numpy as np
from base_model import ModelML
from .tree import TreeNode
from .utils import *

class C50DecisionTree(ModelML):
    def __init__(self, n_estimators=10, min_samples=1):
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self.trees = []
        self.tree_weights = []

    def fit(self, features, labels):
        n_samples = len(labels)
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            tree = self.build_tree(features, labels, weights)
            predictions = np.array([self.predict_node(tree, sample) for sample in features])
            error = np.sum(weights * (predictions != labels)) / np.sum(weights)
            
            if error >= 0.5:
                continue
            
            tree_weight = 0.5 * np.log((1 - error) / max(error, 1e-10))
            self.trees.append(tree)
            self.tree_weights.append(tree_weight)
            
            weights *= np.exp(-tree_weight * labels * (2 * predictions - 1))
            weights /= np.sum(weights)

    def build_tree(self, features, labels, weights):
        best_gain_ratio = 0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        current_entropy = entropy(labels, weights)

        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_labels, true_weights, false_features, false_labels, false_weights = split_data(features, labels, feature, value, weights)
                gain_ratio_value = information_gain(true_labels, false_labels, current_entropy, get_ratio=True, true_weights=true_weights, false_weights=false_weights)

                if gain_ratio_value > best_gain_ratio:
                    best_gain_ratio = gain_ratio_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_labels, true_weights, false_features, false_labels, false_weights)

        if best_gain_ratio > 0:
            true_branch = self.build_tree(best_sets[0], best_sets[1], best_sets[2])
            false_branch = self.build_tree(best_sets[3], best_sets[4], best_sets[5])
            return TreeNode(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch, samples=len(labels))

        return TreeNode(results=np.argmax(np.bincount(labels, weights=weights)), samples=len(labels))

    def predict(self, test_features, test_labels):
        num_samples, _ = test_features.shape

        predictions = np.zeros(num_samples)
        for tree, tree_weight in zip(self.trees, self.tree_weights):
            predictions += tree_weight * np.array([self.predict_node(tree, sample) for sample in test_features])

        final_predictions = np.sign(predictions)
        accuracy, f1 = self.evaluate(final_predictions, test_labels)
        print("Accuracy: {:.5f} F1-score: {:.5f}".format(accuracy, f1))
    
    def predict_node(self, tree, sample):
        if tree.results is not None:
            return tree.results
        else:
            branch = tree.false_branch
            if sample[tree.feature] <= tree.value:
                branch = tree.true_branch
            return self.predict_node(branch, sample)
    
    def __str__(self):
        return "Decision Trees: C5.0/See5 Algorithm"
