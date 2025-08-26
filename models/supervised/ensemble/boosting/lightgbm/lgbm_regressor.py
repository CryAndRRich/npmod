from typing import Optional, Tuple
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
                 gamma: float = 0.0,
                 feature_fraction: float = 1.0,
                 subsample: float = 1.0,
                 random_state: int = 42):
        """
        Initialize the LightGBMRegressor

        Parameters:
            learn_rate: Shrinkage factor applied to each tree's contribution
            n_estimators: Number of boosting rounds (trees to fit)
            max_leaves: Maximum number of leaves in each LightGBMTreeRegressor
            min_samples_split: Minimum samples required to split a leaf
            n_feats: Number of features to consider for each split
            reg_lambda: L2 regularization term on leaf weights
            gamma: Minimum loss reduction required to make a split
            feature_fraction: Fraction of features to consider at each split
            subsample: Fraction of samples to use for fitting each tree
            random_state: Random seed for reproducibility
        """
        self.eta = learn_rate
        self.K = n_estimators
        self.tree_kwargs = dict(
            max_leaves=max_leaves,
            min_samples_split=min_samples_split,
            n_feats=n_feats,
            reg_lambda=reg_lambda,
            gamma=gamma,
            feature_fraction=feature_fraction,
            random_state=random_state
        )
        self.subsample = subsample
        self.random_state = random_state

        self.init_pred = 0.0
        self.trees = []
        self.train_loss_history = []
        self.val_loss_history = []

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray,
            valid_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_rounds: Optional[int] = None) -> None:
        """
        Fit the LightGBMRegressor to the training data

        Parameters:
            features: Training features
            targets: True target values
            valid_set: Optional tuple of (validation_features, validation_targets) for early stopping
            early_stopping_rounds: Number of rounds with no improvement to trigger early stopping
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = features.shape[0]

        self.init_pred = float(np.mean(targets))
        predictions = np.full(n_samples, self.init_pred, dtype=float)

        best_val = np.inf
        best_iter = -1
        rounds_since_best = 0

        self.trees = []
        self.train_loss_history = []
        self.val_loss_history = []

        for m in range(self.K):
            # -optionally subsample rows for this boosting round (bagging)
            if self.subsample < 1.0:
                sample_size = max(1, int(self.subsample * n_samples))
                idx = rng.choice(n_samples, size=sample_size, replace=False)
                _, y_sub = features[idx], targets[idx]
                pred_sub = predictions[idx]
                grad = pred_sub - y_sub
                hess = np.ones_like(grad, dtype=float)
            else:
                idx = np.arange(n_samples)
                grad = predictions - targets
                hess = np.ones_like(targets, dtype=float)

            # fit tree to (grad, hess) using only selected rows
            tree = LightGBMTreeRegressor(**self.tree_kwargs)
            tree.fit(features[idx], grad, hess)

            # tree.predict returns leaf weights (w = -G/(H+lambda) inside each leaf)
            update = tree.predict(features)

            predictions += self.eta * update

            # store tree
            self.trees.append(tree)

            # compute and record training loss (MSE)
            train_mse = float(np.mean((predictions - targets) ** 2))
            self.train_loss_history.append(train_mse)

            # compute validation loss if provided
            val_mse = None
            if valid_set is not None:
                X_val, y_val = valid_set
                val_pred = self.predict(X_val)
                val_mse = float(np.mean((val_pred - y_val) ** 2))
                self.val_loss_history.append(val_mse)
            else:
                self.val_loss_history.append(np.nan)

            # early stopping logic (use validation)
            if valid_set is not None and early_stopping_rounds is not None:
                if val_mse < best_val - 1e-12:
                    best_val = val_mse
                    best_iter = m
                    rounds_since_best = 0
                else:
                    rounds_since_best += 1
                if rounds_since_best >= early_stopping_rounds:
                    # optionally keep only trees up to best_iter
                    self.trees = self.trees[:best_iter+1]
                    self.train_loss_history = self.train_loss_history[:best_iter+1]
                    self.val_loss_history = self.val_loss_history[:best_iter+1]
                    break


    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for given samples

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        preds = np.full(test_features.shape[0], self.init_pred, dtype=float)
        for tree in self.trees:
            preds += self.eta * tree.predict(test_features)
        return preds

    def __str__(self) -> str:
        return "LightGBM Regressor"
