from typing import List
import numpy as np
from .cat_tree import CatTreeRegressor

class CatBoostClassifier():
    """
    CatBoost classifier using gradient boosting
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
        Initialize the CatBoostClassifier

        Parameters:
            learn_rate: Shrinkage factor applied to each tree's output
            n_estimators: Number of boosting rounds
            max_depth: Depth of each oblivious tree
            min_samples_split: Minimum samples to split a node
            n_feats: Number of features to consider at each split
            reg_lambda: L2 regularization term on leaf weights
            gamma: Minimum gain to perform a split
            cat_features: Indices of categorical feature columns
            n_permutations: Permutations for ordered target encoding
            random_seed: Random seed for encoding
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
        self.init_raw = None
        self.trees = []
        self._cat_global_mean = {}
        self.n_classes = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Train CatBoostClassifier on labeled data

        Parameters:
            features: Feature matrix
            targets: Targets values
        """
        n_samples = len(targets)
        self.n_classes = int(np.max(targets)) + 1

        # Optionally store global category 
        for col in self.cat_features:
            vals = features[:, col]
            self._cat_global_mean[col] = {k: np.mean(targets[vals == k]) for k in np.unique(vals)}

        # Initialize raw logits with log priors
        class_counts = np.bincount(targets, minlength=self.n_classes)
        priors = class_counts / float(n_samples)
        priors = np.clip(priors, 1e-12, 1 - 1e-12)
        self.init_raw = np.log(priors)
        raw_pred = np.tile(self.init_raw, (n_samples, 1)) 

        # Boosting loop: each round train one tree per class (one-vs-rest)
        self.trees = []
        for m in range(self.K):
            prob = self._softmax(raw_pred) 
            grad = prob.copy()
            grad[np.arange(n_samples), targets] -= 1.0 
            hess = prob * (1.0 - prob)  # diagonal approx

            trees_for_round = []
            for c in range(self.n_classes):
                tree = CatTreeRegressor(cat_features=self.cat_features, **self.tree_kwargs)
                # pass per-class grad/hess (1D)
                tree.fit(features, grad[:, c], hess[:, c])
                update = tree.predict(features)  # leaf weights
                raw_pred[:, c] += self.eta * update
                trees_for_round.append(tree)

            self.trees.append(trees_for_round)

    def predict_proba(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates for each class

        Parameters:
            test_features: Feature matrix

        Returns:
            np.ndarray: Probabilities for each class
        """
        n = test_features.shape[0]
        raw_pred = np.tile(self.init_raw, (n, 1))
        for trees_for_round in self.trees:
            for c, tree in enumerate(trees_for_round):
                raw_pred[:, c] += self.eta * tree.predict(test_features)
        return self._softmax(raw_pred)
    
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Parameters:
            test_features: Feature matrix

        Returns:
            np.ndarray: Predicted labels
        """
        proba = self.predict_proba(test_features)
        return np.argmax(proba, axis=1)

    def __str__(self) -> str:
        return "CatBoost Classifier"
