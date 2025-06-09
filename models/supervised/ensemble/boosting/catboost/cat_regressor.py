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
                 n_permutations: int = 1,
                 random_seed: int = 42) -> None:
        """
        Initialize the CatBoostRegressor

        Parameters:
            learn_rate: Shrinkage factor applied to each tree's output
            n_estimators: Number of boosting rounds
            max_depth: Depth of each oblivious CatTreeRegressor
            min_samples_split: Minimum samples to split a node
            n_feats: Number of features to consider at each split
            reg_lambda: L2 regularization term (λ) on leaf weights
            gamma: Minimum gain (γ) to perform a split
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
        self.n_permutations = n_permutations
        self.random_seed = random_seed
        self.init_pred = None
        self.trees = []
        # Store mapping for unseen data
        self._cat_global_mean  = {}

    def ordered_target_encoding(self, 
                                feature_cols: np.ndarray, 
                                targets: np.ndarray) -> np.ndarray:
        """
        Perform ordered target encoding on a categorical column

        Parameters:
            feature_cols: Categorical values
            targets: Continuous target array

        Returns:
            np.ndarray: Encoded numeric array
        """
        n = len(targets)
        encoded = np.zeros(n, dtype=float)
        rng = np.random.RandomState(self.random_seed)
        for _ in range(self.n_permutations):
            perm = rng.permutation(n)
            sums = {}
            counts = {}
            tmp = np.zeros(n, dtype=float)
            for idx in perm:
                key = feature_cols[idx]
                if counts.get(key, 0) > 0:
                    tmp[idx] = sums[key] / counts[key]
                else:
                    tmp[idx] = np.mean(targets[:idx]) if idx > 0 else 0.0
                sums[key] = sums.get(key, 0.0) + targets[idx]
                counts[key] = counts.get(key, 0) + 1
            encoded += tmp
        return encoded / self.n_permutations

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Train CatBoostRegressor on data

        Parameters:
            features: Feature matrix, shape (n_samples, n_features)
            targets: Targets, shape (n_samples,)
        """
        # Convert features and perform encoding
        features_enc = features.copy()
        for col in self.cat_features:
            features_enc[:, col] = self.ordered_target_encoding(features[:, col], targets)
        features_enc = features_enc.astype(float)

        # Store global means for unseen data
        for col in self.cat_features:
            vals = features[:, col]
            self._cat_global_mean[col] = {
                k: np.mean(targets[vals == k]) for k in np.unique(vals)
            }

        # Initialize
        self.init_pred = np.mean(targets)
        predictions = np.full_like(targets, self.init_pred, dtype=float)

        # Boosting loop
        for _ in range(self.K):
            grad = predictions - targets
            hess = np.ones_like(targets)
            tree = CatTreeRegressor(
                cat_features=self.cat_features,
                **self.tree_kwargs
            )
            tree.fit(features_enc, grad, hess)
            update = tree.predict(features_enc)
            predictions -= self.eta * update
            self.trees.append(tree)

    def _global_encoding(self, features: np.ndarray) -> np.ndarray:
        """
        Encode categorical columns using stored global means for any unseen categories
        """
        features_enc = features.copy().astype(object)
        for col, mapping in self._cat_global_mean.items():
            col_vals = features[:, col]
            enc = np.array([
                mapping.get(v, np.mean(list(mapping.values())))
                for v in col_vals
            ], dtype=float)
            features_enc[:, col] = enc
        return features_enc.astype(float)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous outputs for input data

        Parameters:
            test_features: Feature matrix

        Returns:
            np.ndarray: Predicted values
        """
        features_enc = self._global_encoding(test_features)
        predictions = np.full(shape=(test_features.shape[0],), fill_value=self.init_pred, dtype=float)
        for tree in self.trees:
            predictions -= self.eta * tree.predict(features_enc)

        return predictions

    def __str__(self) -> str:
        return "CatBoost Regressor"
