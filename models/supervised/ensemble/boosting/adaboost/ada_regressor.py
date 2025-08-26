import numpy as np
from ....regression import DecisionTreeRegressor

class AdaBoostRegressor():
    def __init__(self,
                 learn_rate: float = 1.0,
                 number_of_epochs: int = 50,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 n_feats: int = None) -> None:
        """
        AdaBoost Regressor using custom DecisionTreeRegressor as weak learner
        following the AdaBoost.R2 algorithm

        Parameters:
            learn_rate: Shrinkage factor applied to each weak learner's weight
            number_of_epochs: Number of boosting rounds
            max_depth: Maximum depth of each regression tree (weak learner)
            min_samples_split: Minimum samples to split an internal node
            n_feats: Number of features to consider when looking for best split
            random_state: Seed for reproducibility
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_feats = n_feats

        self.trees = []
        self.alphas = []
        self.base_pred = None

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Train the AdaBoost regressor

        Parameters:
            features: Training feature matrix
            targets: Continuous target vector
        """
        n_samples = features.shape[0]

        sample_weights = np.full(n_samples, 1.0 / n_samples, dtype=float)
        self.base_pred = np.mean(targets) 

        for _ in range(self.number_of_epochs):
            tree = DecisionTreeRegressor(
                n_feats=self.n_feats,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(features, targets, sample_weights)

            preds = tree.predict(features)
            errors = np.abs(preds - targets)
            max_error = np.max(errors)

            if max_error < 1e-12:
                self.trees.append(tree)
                self.alphas.append(1.0) 
                break

            normalized_errors = errors / max_error
            err_m = np.dot(sample_weights, normalized_errors)

            if err_m >= 0.5:
                continue

            beta_m = err_m / max(1.0 - err_m, 1e-12)
            alpha_m = self.learn_rate * np.log(1.0 / (beta_m + 1e-12))

            sample_weights *= np.power(beta_m, (1.0 - normalized_errors))
            sample_weights /= np.sum(sample_weights)

            self.trees.append(tree)
            self.alphas.append(alpha_m)

        if not self.trees:
            self.trees = []
            self.alphas = []

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values using the trained AdaBoost model

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted continuous values
        """
        if not self.trees:
            return np.full(test_features.shape[0], self.base_pred, dtype=float)

        agg = np.zeros(test_features.shape[0], dtype=float)
        for tree, alpha in zip(self.trees, self.alphas):
            agg += alpha * tree.predict(test_features)

        return agg / (np.sum(self.alphas) + 1e-12)

    def __str__(self) -> str:
        return "AdaBoost Regressor"
