import numpy as np
from .xgb_tree import XGTreeRegressor

class XGBClassifier():
    """
    XGBoost-style classifier implemented from scratch using second-order approximation
    """
    def __init__(self,
                 n_classes: int,
                 learn_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 n_feats: int = None,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0) -> None:
        """
        Initialize the XGBClassifier

        Parameters:
            n_classes: Number of classes
            learn_rate: Shrinkage factor applied to each tree's contribution
            n_estimators: Number of boosting rounds (trees to fit)
            max_depth: Maximum depth of each XGTreeRegressor
            min_samples_split: Minimum samples required to split an internal node
            n_feats: Number of features to consider for each split
            reg_lambda: L2 regularization term on leaf weights
            gamma: Minimum loss reduction required to make a split
        """
        self.n_classes = n_classes
        self.learn_rate = learn_rate
        self.n_estimators = n_estimators
        self.tree_kwargs = dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_feats=n_feats,
            reg_lambda=reg_lambda,
            gamma=gamma
        )
        self.init_raw = None
        self.trees = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities row-wise"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Fit the XGBClassifier

        Parameters:
            features: Training features
            targets: Target values
        """
        n_samples = features.shape[0]
        self.init_raw = np.zeros(self.n_classes, dtype=float)
        raw_pred = np.zeros((n_samples, self.n_classes), dtype=float)
        self.trees = [[None]*self.n_estimators for _ in range(self.n_classes)]

        # Boosting rounds
        for t in range(self.n_estimators):
            prob = self._softmax(raw_pred)
            for k in range(self.n_classes):
                yk = (targets == k).astype(float)
                grad = prob[:, k] - yk           # Gradient
                hess = prob[:, k] * (1 - prob[:, k])  # Hessian
                tree = XGTreeRegressor(**self.tree_kwargs)
                tree.fit(features, grad, hess)
                update = tree.predict(features)
                # Correct update sign
                raw_pred[:, k] += self.learn_rate * update
                self.trees[k][t] = tree

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples

        Parameters:
            features: Feature matrix

        Returns:
            np.ndarray: Array of probabilities for the positive class
        """
        n_samples = features.shape[0]
        raw_pred = np.zeros((n_samples, self.n_classes), dtype=float)
        for k in range(self.n_classes):
            for t in range(self.n_estimators):
                raw_pred[:, k] += self.learn_rate * self.trees[k][t].predict(features)
        return self._softmax(raw_pred)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples

        Parameters:
            test_features: Feature matrix

        Returns:
            np.ndarray: Predicted class labels
        """
        prob = self.predict_proba(test_features)
        return np.argmax(prob, axis=1)

    def __str__(self) -> str:
        return "XGBoost Classifier"
