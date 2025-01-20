import numpy as np
from base_model import ModelML

class TreeNode:
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None, samples=None, chi_square=None):
        self.feature = feature
        self.value = value
        self.results = results
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.samples = samples
        self.chi_square = chi_square

class Tree(ModelML):
    def fit(self, features, labels):
        self.decision_tree = self.build_tree(features, labels)
    
    def build_tree(self, features, labels):
        pass

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
    
