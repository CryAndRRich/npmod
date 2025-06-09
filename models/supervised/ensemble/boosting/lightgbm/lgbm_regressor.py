import numpy as np
from .lgbm_tree import LightGBMTreeRegressor

class LightGBMRegressor():
    """
    LightGBM-style regressor implemented from scratch using leaf-wise tree growth.
    Utilizes LightGBMTreeRegressor as the base learner with squared error objective
    """
    def __init__(self,
                 learn_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_leaves: int = 31,
                 min_samples_split: int = 20,
                 n_feats: int = None,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0) -> None:
        """
        Initialize the LightGBMRegressor

        Parameters:
            learn_rate: Shrinkage factor (η) applied to each tree's contribution
            n_estimators: Number of boosting rounds (trees to fit)
            max_leaves: Maximum number of leaves in each LightGBMTreeRegressor
            min_samples_split: Minimum samples required to split a leaf
            n_feats: Number of features to consider for each split
            reg_lambda: L2 regularization term on leaf weights (λ)
            gamma: Minimum loss reduction required to make a split (γ)
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
        self.init_pred = 0.0
        self.trees = []

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Fit the LightGBMRegressor to the training data.
        Uses squared error objective: loss = 0.5 * (targets - f)^2

        Parameters:
            features: Training features, shape (n_samples, n_features)
            targets: True target values, shape (n_samples,)
        """
        # Initial prediction is mean of targets
        self.init_pred = float(np.mean(targets))
        predictions = np.full_like(targets, fill_value=self.init_pred, dtype=float)

        for _ in range(self.K):
            grad = predictions - targets
            hess = np.ones_like(targets, dtype=float)

            tree = LightGBMTreeRegressor(**self.tree_kwargs)
            tree.fit(features, grad, hess)
            update = tree.predict(features)

            # Update ensemble prediction
            predictions -= self.eta * update
            self.trees.append(tree)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for given samples

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        # Start from the initial constant prediction
        predictions = np.full(shape=(test_features.shape[0],), fill_value=self.init_pred, dtype=float)

        # Aggregate contributions from each tree
        for tree in self.trees:
            predictions -= self.eta * tree.predict(test_features)

        return predictions

    def __str__(self) -> str:
        return "LightGBM Regressor"
