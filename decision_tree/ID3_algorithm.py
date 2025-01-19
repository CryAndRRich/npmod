import numpy as np
from base_model import ModelML
from .tree import TreeNode
from .utils import *

class ID3DecisionTree(ModelML):
    def fit(self, features, labels):
        self.decision_tree = self.build_tree(features, labels)
    
    def build_tree(self, features, labels):
        best_gain = 0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        current_entropy = entropy(labels)

        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_labels, false_features, false_labels = split_data(features, labels, feature, value)
                information_gain_value = information_gain(true_labels, false_labels, current_entropy)

                if information_gain_value > best_gain:
                    best_gain = information_gain_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_labels, false_features, false_labels)

        if best_gain > 0:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)

        return TreeNode(results=labels[0])

    def predict(self, test_features, test_labels):
        num_samples, _ = test_features.shape

        predictions = np.zeros(num_samples)
        for ind, feature in enumerate(test_features):
            predictions[ind] = self.predict_node(self.decision_tree, feature)

        accuracy, f1 = self.evaluate(predictions, test_labels)
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
        return "Decision Trees: ID3 Algorithm"