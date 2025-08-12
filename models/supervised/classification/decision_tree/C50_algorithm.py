import numpy as np
from .tree import *
from .utils import entropy, information_gain, split_data

class C50DecisionTree(Tree):
    def __init__(self, 
		 n_estimators: int = 10) -> None:
        self.n_estimators = n_estimators
        self.trees = []
        self.tree_weights = []
        self.classes_ = None

    def c45_train(self,
                  X: np.ndarray,
                  y: np.ndarray,
                  weights: np.ndarray = None,
                  min_samples: int = 2) -> TreeNode:

        if len(np.unique(y)) == 1:
            return TreeNode(results=np.unique(y)[0])

        if len(y) < min_samples:
            most_common = Counter(y).most_common(1)[0][0]
            return TreeNode(results=most_common)

        current_entropy = entropy(y, weights)
        best_gain_ratio = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            thresholds = ((values[:-1] + values[1:]) / 2) if len(values) > 10 else values

            for threshold in thresholds:
                # Split
                if weights is None:
                    X_true, y_true, X_false, y_false = split_data(X, y, feature, threshold)
                    true_w = false_w = None
                else:
                    X_true, y_true, true_w, X_false, y_false, false_w = split_data(X, y, feature, threshold, weights)

                if len(y_true) == 0 or len(y_false) == 0:
                    continue

                gain_ratio = information_gain(
                    y_true, y_false, current_entropy,
                    true_weights=true_w,
                    false_weights=false_w,
                    get_ratio=True
                )

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = threshold

        if best_gain_ratio <= 0:
            most_common = Counter(y).most_common(1)[0][0]
            return TreeNode(results=most_common)

        if weights is None:
            X_true, y_true, X_false, y_false = split_data(X, y, best_feature, best_threshold)
            true_w = false_w = None
        else:
            X_true, y_true, true_w, X_false, y_false, false_w = split_data(X, y, best_feature, best_threshold, weights)

        true_branch = self.c45_train(X_true, y_true, true_w, min_samples)
        false_branch = self.c45_train(X_false, y_false, false_w, min_samples)

        return TreeNode(feature=best_feature, value=best_threshold,
                        true_branch=true_branch, false_branch=false_branch)

    def c45_predict(self,
                    tree: TreeNode, 
                    x: np.ndarray) -> int:
        if tree.results is not None:
            return tree.results
    
        if isinstance(tree.value, (int, float)):
            branch = tree.true_branch if x[tree.feature] <= tree.value else tree.false_branch
        else:
            branch = tree.true_branch if x[tree.feature] == tree.value else tree.false_branch
        
        return self.c45_predict(branch, x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.classes_ = np.unique(y)
        y_encoded = np.where(y == self.classes_[0], -1, 1)
        n_samples = len(y)
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            tree = self.c45_train(X, y, weights)
            preds = np.array([self.c45_predict(tree, xi) for xi in X])
            preds_encoded = np.where(preds == self.classes_[0], -1, 1)

            err = np.sum(weights[preds_encoded != y_encoded])
            err = max(err, 1e-10)  
            alpha = 0.5 * np.log((1 - err) / err)
 
            weights *= np.exp(-alpha * y_encoded * preds_encoded)
            weights /= np.sum(weights)

            self.trees.append(tree)
            self.tree_weights.append(alpha)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds_sum = np.zeros(X.shape[0])
        for alpha, tree in zip(self.tree_weights, self.trees):
            preds = np.array([self.c45_predict(tree, xi) for xi in X])
            preds_encoded = np.where(preds == self.classes_[0], -1, 1)
            preds_sum += alpha * preds_encoded
        return np.where(preds_sum >= 0, self.classes_[1], self.classes_[0])
    
    def __str__(self) -> str:
        return "C5.0/See5 Algorithm"
