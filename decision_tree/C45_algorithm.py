import numpy as np
from .tree import *
from .utils import *

class C45DecisionTree(Tree):
    def build_tree(self, features, labels):
        best_gain_ratio = 0
        best_criteria = None
        best_sets = None
        _, n = features.shape

        current_entropy = entropy(labels)

        for feature in range(n):
            feature_values = set(features[:, feature])
            for value in feature_values:
                true_features, true_labels, false_features, false_labels = split_data(features, labels, feature, value)
                gain_ratio_value = information_gain(true_labels, false_labels, current_entropy, get_ratio=True)

                if gain_ratio_value > best_gain_ratio:
                    best_gain_ratio = gain_ratio_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_labels, false_features, false_labels)

        if best_gain_ratio > 0:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return TreeNode(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)

        return TreeNode(results=np.argmax(np.bincount(labels)))

    def __str__(self):
        return "Decision Trees: C4.5 Algorithm"
