from typing import List
import numpy as np
from .cat_tree import CatTreeRegressor

class CatBoostRegressor():
    """
    CatBoost gradient boosting regressor with native categorical support.
    Utilizes ordered target encoding and oblivious CatTreeRegressor for boosting
    """
    def __init__(self,
                 learn_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 n_feats: int = None,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0,
                 cat_features: List[int] = None,
                 random_seed: int = 42) -> None:
        """
        Initialize the CatBoostRegressor

        Parameters:
            learn_rate: Shrinkage factor applied to each tree's output
            n_estimators: Number of boosting rounds
            max_depth: Depth of each oblivious CatTreeRegressor
            min_samples_split: Minimum samples to split a node
            n_feats: Number of features to consider at each split
            reg_lambda: L2 regularization term on leaf weights
            gamma: Minimum gain to perform a split
            cat_features: Indices of categorical feature columns
            n_permutations: Permutations for ordered target encoding
            random_seed: Random seed for encoding permutations
        """
        self.eta = learn_rate
        self.K = n_estimators
        self.tree_kwargs = dict(
            n_feats=n_feats,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            reg_lambda=reg_lambda,
            gamma=gamma
        )
        self.cat_features = cat_features or []
        self.random_seed = random_seed
        self.init_pred = None
        self.trees = []
        self._cat_global_mean = {}

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Train CatBoostRegressor on data

        Parameters:
            features: Feature matrix
            targets: Targets values
        """
        # store global means (optional, can still keep for other purposes)
        for col in self.cat_features:
            vals = features[:, col]
            self._cat_global_mean[col] = {
                k: np.mean(targets[vals == k]) for k in np.unique(vals)
            }

        self.init_pred = np.mean(targets)
        preds = np.full_like(targets, self.init_pred, dtype=float)

        for _ in range(self.K):
            grad = preds - targets
            hess = np.ones_like(targets)

            tree = CatTreeRegressor(
                cat_features=self.cat_features,
                **self.tree_kwargs
            )

            tree.fit(features, grad, hess)
            update = tree.predict(features)

            # update predictions by adding tree's output
            preds += self.eta * update

            self.trees.append(tree)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous outputs for input data

        Parameters:
            test_features: Feature matrix

        Returns:
            np.ndarray: Predicted values
        """
        preds = np.full(test_features.shape[0], self.init_pred, dtype=float)
        for tree in self.trees:
            preds += self.eta * tree.predict(test_features)
        return preds

    def __str__(self) -> str:
        return "CatBoost Regressor"
