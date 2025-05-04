from typing import List
import numpy as np
from .....base import Model
from .cat_tree import CatTreeRegressor

class CatBoostClassifier(Model):
    """
    CatBoost binary classifier using gradient boosting
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
                 random_seed: int = 42,
                 threshold: float = 0.5) -> None:
        """
        Initialize the CatBoostClassifier

        Parameters:
            learn_rate: Shrinkage factor applied to each tree's output
            n_estimators: Number of boosting rounds
            max_depth: Depth of each oblivious tree
            min_samples_split: Minimum samples to split a node
            n_feats: Number of features to consider at each split
            reg_lambda: L2 regularization term (λ) on leaf weights
            gamma: Minimum gain (γ) to perform a split
            cat_features: Indices of categorical feature columns
            n_permutations: Permutations for ordered target encoding
            random_seed: Random seed for encoding
            threshold: Probability threshold for binary classification
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
        self.threshold = threshold
        self.init_raw = None
        self.trees = []
        self._cat_global_mean = {}

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation

        Parameters:
            x: Raw score array

        Returns:
            np.ndarray: Sigmoid probabilities
        """
        return 1.0 / (1.0 + np.exp(-x))

    def ordered_target_encoding(self, 
                                feature_cols: np.ndarray, 
                                targets: np.ndarray) -> np.ndarray:
        """
        Perform ordered target encoding for categorical column

        Parameters:
            feature_cols: Categorical values
            targets: Binary target values (0 or 1)

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
            temp = np.zeros(n, dtype=float)
            for idx in perm:
                key = feature_cols[idx]
                if counts.get(key, 0) > 0:
                    temp[idx] = sums[key] / counts[key]
                else:
                    temp[idx] = np.mean(targets[:idx]) if idx > 0 else 0.0
                sums[key] = sums.get(key, 0.0) + targets[idx]
                counts[key] = counts.get(key, 0) + 1
            encoded += temp
        return encoded / self.n_permutations

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Train CatBoostClassifier on binary-labeled data.

        Parameters:
            features: Feature matrix, shape (n_samples, n_features)
            targets: Targets, shape (n_samples,)
        """
        # Encode categorical
        features_enc = features.copy()
        for col in self.cat_features:
            features_enc[:, col] = self.ordered_target_encoding(features[:, col], targets)
        features_enc = features_enc.astype(float)

        # Store global means for unseen
        for col in self.cat_features:
            vals = features[:, col]
            self._cat_global_mean[col] = {k: np.mean(targets[vals == k]) for k in np.unique(vals)}

        # Initialize raw score = log-odds(p)
        p = np.clip(np.mean(targets), 1e-6, 1 - 1e-6)
        self.init_raw = np.log(p / (1 - p))
        raw_pred = np.full_like(targets, fill_value=self.init_raw, dtype=float)

        # Boosting iterations
        for _ in range(self.K):
            prob = self._sigmoid(raw_pred)
            grad = prob - targets
            hess = prob * (1 - prob)

            tree = CatTreeRegressor(
                cat_features=self.cat_features,
                **self.tree_kwargs
            )
            tree.fit(features_enc, grad, hess)
            update = tree.predict(features_enc)

            raw_pred -= self.eta * update
            self.trees.append(tree)

    def _global_encoding(self, features: np.ndarray) -> np.ndarray:
        """
        Apply stored encoding for unseen data
        """
        features_enc = features.copy().astype(object)
        for col, mapping in self._cat_global_mean.items():
            col_vals = features[:, col]
            enc = np.array([mapping.get(v, np.mean(list(mapping.values()))) for v in col_vals], dtype=float)
            features_enc[:, col] = enc
        return features_enc.astype(float)

    def predict_proba(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict probability of positive class

        Parameters:
            test_features: Feature matrix

        Returns:
            np.ndarray: Probabilities for class 1
        """
        features_enc = self._global_encoding(test_features)
        raw_pred = np.full(shape=(test_features.shape[0],), fill_value=self.init_raw, dtype=float)
        for tree in self.trees:
            raw_pred -= self.eta * tree.predict(features_enc)
        return self._sigmoid(raw_pred)

    def predict(self, 
                test_features: np.ndarray, 
                test_targets: np.ndarray = None) -> np.ndarray:
        """
        Predict binary class labels

        Parameters:
            test_features: Feature matrix
            test_targets: True labels for evaluation

        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        proba = self.predict_proba(test_features)
        predictions = (proba >= self.threshold).astype(int)

        if test_targets is not None:
            acc = np.mean(predictions == test_targets)
            print(f"Accuracy: {acc:.5f}")

        return predictions

    def __str__(self) -> str:
        return "CatBoost Classifier"
