import numpy as np
from .lgbm_tree import LightGBMTreeRegressor

class LightGBMClassifier():
    """
    LightGBM-style binary classifier implemented from scratch using leaf-wise tree growth.
    Utilizes LightGBMTreeRegressor as the base learner with logistic loss
    """
    def __init__(self,
                 learn_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_leaves: int = 31,
                 min_samples_split: int = 20,
                 n_feats: int = None,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0,
                 threshold: float = 0.5) -> None:
        """
        Initialize the LightGBMClassifier

        Parameters:
            learn_rate: Shrinkage factor (η) applied to each tree's contribution
            n_estimators: Number of boosting rounds (trees to fit)
            max_leaves: Maximum number of leaves in each LightGBMTreeRegressor
            min_samples_split: Minimum samples required to split a leaf
            n_feats: Number of features to consider for each split
            reg_lambda: L2 regularization term on leaf weights (λ)
            gamma: Minimum loss reduction required to make a split (γ)
            threshold: Probability threshold for converting probabilities to class labels
        """
        self.eta = learn_rate
        self.K = n_estimators
        self.tree_kwargs = dict(
            max_leaves=max_leaves,
            min_samples_split=min_samples_split,
            n_feats=n_feats,
            reg_lambda=reg_lambda,
            gamma=gamma
        )
        # Initial raw score (log_odds)
        self.init_raw_score = 0.0
        self.trees = []
        self.threshold = threshold

    def fit(self,
            features: np.ndarray,
            labels: np.ndarray) -> None:
        """
        Fit the LightGBMClassifier to the training data.
        Uses logistic loss for binary classification

        Parameters:
            features: Training features, shape (n_samples, n_features)
            labels: Binary target values (0 or 1), shape (n_samples,)
        """
        # Initialize raw scores to log odds of positive class
        p = np.clip(np.mean(labels), 1e-6, 1 - 1e-6)
        self.init_raw_score = float(np.log(p / (1 - p)))
        raw_scores = np.full_like(labels, fill_value=self.init_raw_score, dtype=float)

        for _ in range(self.K):
            # Compute predictions and probabilities
            probs = 1 / (1 + np.exp(-raw_scores))

            # Gradient and hessian for logistic loss
            grad = probs - labels
            hess = probs * (1 - probs)

            tree = LightGBMTreeRegressor(**self.tree_kwargs)
            tree.fit(features, grad, hess)
            update = tree.predict(features)

            # Update raw scores
            raw_scores -= self.eta * update
            self.trees.append(tree)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for given samples

        Parameters:
            features: Feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Array of shape (n_samples, 2)
        """
        raw_scores = np.full(shape=(features.shape[0],), fill_value=self.init_raw_score, dtype=float)
        for tree in self.trees:
            raw_scores -= self.eta * tree.predict(features)

        probs_pos = 1 / (1 + np.exp(-raw_scores))
        probs_neg = 1 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input samples

        Parameters:
            test_features: Feature matrix

        Returns:
            np.ndarray: Predicted class labels (0 or 1)
        """
        prob = self.predict_proba(test_features)[:, 1]
        predictions = (prob >= self.threshold).astype(int)

        return predictions

    def __str__(self) -> str:
        return "LightGBM Classifier"
