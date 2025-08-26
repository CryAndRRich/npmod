from typing import Optional, Tuple
import numpy as np
from .lgbm_tree import LightGBMTreeRegressor

class LightGBMClassifier():
    """
    LightGBM-style classifier implemented from scratch using leaf-wise tree growth
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
                 random_state: int = 42,
                 eps: float = 1e-12) -> None:
        """
        Initialize the LightGBMClassifier

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
            eps: Small constant to avoid numerical issues
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
        self.eps = eps

        self.n_classes_ = None
        self.init_raw_ = None   
        self.trees = []  

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x - np.max(x, axis=1, keepdims=True)
        expz = np.exp(z)
        denom = np.sum(expz, axis=1, keepdims=True)
        return expz / (denom + self.eps)
    
    def fit(self,
            features: np.ndarray,
            labels: np.ndarray,
            valid_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_rounds: Optional[int] = None) -> None:
        """
        Fit the LightGBMClassifier to the training data

        Parameters:
            features: Training features
            labels: Target values 
            valid_set: Optional validation set for early stopping
            early_stopping_rounds: Rounds with no improvement to trigger early stopping
        """
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(features)
        y = np.asarray(labels).astype(int)
        n_samples, _ = X.shape

        classes, _ = np.unique(y, return_inverse=True)
        K = len(classes)
        self.n_classes_ = K

        # initial raw logits
        class_counts = np.bincount(y, minlength=K)
        priors = class_counts / float(n_samples)
        priors = np.clip(priors, self.eps, 1.0 - self.eps)
        self.init_raw_ = np.log(priors)

        # raw scores per sample per class
        raw = np.tile(self.init_raw_, (n_samples, 1)).astype(float)

        # bookkeeping
        self.trees = []
        best_val = np.inf
        rounds_since_best = 0
        best_iter = -1

        for m in range(self.K):
            # compute probabilities and per-class grad/hess
            prob = self._softmax(raw) 
            # g = p - y_onehot
            g = prob.copy()
            g[np.arange(n_samples), y] -= 1.0
            # hessian diagonal approx
            h = prob * (1.0 - prob)

            # optional subsample rows
            if self.subsample < 1.0:
                sample_size = max(1, int(self.subsample * n_samples))
                rows = rng.choice(n_samples, size=sample_size, replace=False)
            else:
                rows = np.arange(n_samples)

            trees_this_round = []
            # train one tree per class
            for c in range(K):
                grad_c = g[rows, c]
                hess_c = h[rows, c]

                grad_c = np.clip(grad_c, -1e6, 1e6)
                hess_c = np.clip(hess_c, self.eps, 1e6)

                tree = LightGBMTreeRegressor(**self.tree_kwargs)
                # fit on the sampled rows only (tree code expects features, grad, hess arrays aligned)
                tree.fit(X[rows], grad_c, hess_c)

                # predict leaf weights for all samples
                update_c = tree.predict(X)
                # ensure finite and clip extreme updates
                update_c = np.clip(update_c, -1e6, 1e6)

                raw[:, c] += self.eta * update_c

                trees_this_round.append(tree)

            self.trees.append(trees_this_round)

            # early stopping if validation given
            if valid_set is not None and early_stopping_rounds is not None:
                X_val, y_val = valid_set
                probs_val = self.predict_proba(X_val)
                probs_true_val = np.clip(probs_val[np.arange(len(y_val)), y_val], self.eps, 1.0)
                val_logloss = -np.mean(np.log(probs_true_val))

                if val_logloss + 1e-12 < best_val:
                    best_val = val_logloss
                    best_iter = m
                    rounds_since_best = 0
                else:
                    rounds_since_best += 1

                if rounds_since_best >= early_stopping_rounds:
                    self.trees = self.trees[:best_iter+1]
                    break

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for given samples

        Parameters:
            features: Feature matrix
        
        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        X = np.asarray(features)
        n = X.shape[0]
        # start from init logits
        raw = np.tile(self.init_raw_, (n, 1)).astype(float)
        for trees_round in self.trees:
            for c, tree in enumerate(trees_round):
                # add tree contribution
                raw[:, c] += self.eta * tree.predict(X)
        proba = self._softmax(raw)
        return proba

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples

        Parameters:
            test_features: Feature matrix

        Returns:
            np.ndarray: Predicted class labels
        """
        proba = self.predict_proba(test_features)
        return np.argmax(proba, axis=1)

    def __str__(self) -> str:
        return "LightGBM Classifier"
