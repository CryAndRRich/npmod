import numpy as np
from .xgb_tree import XGTreeRegressor

class XGBRegressor():
    """
    XGBoost-style regressor implemented from scratch using second-order approximation.
    Utilizes XGTreeRegressor as the base learner with squared error objective
    """
    def __init__(self,
                 learn_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 n_feats: int = None,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0) -> None:
        """
        Initialize the XGBRegressor

        Parameters:
            learn_rate: Shrinkage factor (η) applied to each tree's contribution
            n_estimators: Number of boosting rounds (trees to fit)
            max_depth: Maximum depth of each XGTreeRegressor
            min_samples_split: Minimum samples required to split an internal node
            n_feats: Number of features to consider for each split
            reg_lambda: L2 regularization term on leaf weights (λ)
            gamma: Minimum loss reduction required to make a split (γ)
        """
        self.learn_rate = learn_rate
        self.n_estimators = n_estimators
        self.tree_kwargs = dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_feats=n_feats,
            reg_lambda=reg_lambda,
            gamma=gamma
        )
        self.init_pred = None
        self.trees = []

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fit the XGBRegressor to the training data.
        Uses squared error objective: loss = 0.5 * (targets - f)^2

        Parameters:
            features: Training features, shape (n_samples, n_features)
            targets: True target values, shape (n_samples,)
        """
        # Initial prediction: mean of targets
        self.init_pred = np.mean(targets)
        predictions = np.full_like(targets, fill_value=self.init_pred, dtype=float)

        for _ in range(self.n_estimators):
            # Squared error gradient and hessian
            grad = predictions - targets
            hess = np.ones_like(targets)

            # Fit tree
            tree = XGTreeRegressor(**self.tree_kwargs)
            tree.fit(features, grad, hess)
            update = tree.predict(features)

            # Update predictions with correct sign
            predictions += self.learn_rate * update

            self.trees.append(tree)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for given samples

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        predictions = np.full(shape=(test_features.shape[0],), fill_value=self.init_pred, dtype=float)
        for tree in self.trees:
            predictions += self.learn_rate * tree.predict(test_features)
        return predictions
    
    def __str__(self) -> str:
        return "XGBoost Regressor"
