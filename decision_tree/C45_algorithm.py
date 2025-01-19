import numpy as np
from base_model import ModelML
from .tree import TreeNode
from .utils import *

class C45DecisionTree(ModelML):
    def fit(self, features, labels):
        self.decision_tree = self.build_tree(features, labels)
    
    def build_tree(self, features, labels):
        best_gain_ratio = 0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        current_entropy = entropy(labels)

        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_label, false_feature, false_label = split_data(features, labels, feature, value)
                gain_ratio_value = gain_ratio(true_label, false_label, current_entropy)

                if gain_ratio_value > best_gain_ratio:
                    best_gain_ratio = gain_ratio_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_label, false_feature, false_label)

        if best_gain_ratio > 0:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)

        return TreeNode(results=np.argmax(np.bincount(labels)))

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
            if sample[tree.feature] <= tree.value if isinstance(tree.value, (int, float)) else sample[tree.feature] == tree.value:
                branch = tree.true_branch
            return self.predict_node(branch, sample)
    
    def __str__(self):
        return "Decision Trees: C4.5 Algorithm"
