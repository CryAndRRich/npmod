import numpy as np
from .xgb_tree import XGTreeRegressor
from .....base import Model

class XGBClassifier(Model):
    """
    XGBoost-style classifier implemented from scratch using second-order approximation.
    Utilizes XGTreeRegressor as the base learner with logistic loss for binary classification
    """
    def __init__(self,
                 learn_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 n_feats: int = None,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0,
                 threshold: float = 0.5) -> None:
        """
        Initialize the XGBClassifier

        Parameters:
            learn_rate: Shrinkage factor (η) applied to each tree's contribution
            n_estimators: Number of boosting rounds (trees to fit)
            max_depth: Maximum depth of each XGTreeRegressor
            min_samples_split: Minimum samples required to split an internal node
            n_feats: Number of features to consider for each split
            reg_lambda: L2 regularization term on leaf weights (λ)
            gamma: Minimum loss reduction required to make a split (γ)
            threshold: Probability threshold for converting probabilities to class labels
        """
        self.eta = learn_rate
        self.K = n_estimators
        self.tree_kwargs = dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_feats=n_feats,
            reg_lambda=reg_lambda,
            gamma=gamma
        )
        self.threshold = threshold
        self.init_raw = None
        self.trees = []

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function

        Parameters:
            x: Input array of raw scores

        Returns:
            np.ndarray: Sigmoid probabilities
        """
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Fit the XGBClassifier to binary-labeled data using logistic loss

        Parameters:
            features: Training features, shape (n_samples, n_features)
            targets: Binary target values (0 or 1), shape (n_samples,)
        """
        # Initialize raw predictions to log-odds of positive class
        p = np.clip(np.mean(targets), 1e-6, 1 - 1e-6)
        self.init_raw = np.log(p / (1 - p))
        raw_pred = np.full_like(targets, fill_value=self.init_raw, dtype=float)

        # Boosting iterations
        for _ in range(self.K):
            prob = self._sigmoid(raw_pred)
            grad = prob - targets
            hess = prob * (1 - prob)
            tree = XGTreeRegressor(**self.tree_kwargs)
            tree.fit(features, grad, hess)
            update = tree.predict(features)
            raw_pred -= self.eta * update
            self.trees.append(tree)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Parameters:
            features: Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Array of probabilities for the positive class, shape (n_samples,)
        """
        # Start from initial raw score
        raw_pred = np.full(
            shape=(features.shape[0],),
            fill_value=self.init_raw,
            dtype=float
        )
        # Add tree contributions
        for tree in self.trees:
            raw_pred -= self.eta * tree.predict(features)
        return self._sigmoid(raw_pred)

    def predict(self,
                test_features: np.ndarray,
                test_targets: np.ndarray = None) -> np.ndarray:
        """
        Predict binary class labels for input samples

        Parameters:
            test_features: Feature matrix, shape (n_samples, n_features)
            test_targets: True labels for evaluation

        Returns:
            np.ndarray: Predicted class labels (0 or 1)
        """
        prob = self.predict_proba(test_features)
        predictions = (prob >= self.threshold).astype(int)

        if test_targets is not None:
            accuracy, f1 = self.classification_evaluate(predictions, test_targets)
            print("Accuracy: {:.5f} F1-score: {:.5f}".format(accuracy, f1))

        return predictions

    def __str__(self) -> str:
        return "XGBoost Classifier"
