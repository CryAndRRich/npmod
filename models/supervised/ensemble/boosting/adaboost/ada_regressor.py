import numpy as np
from .....base import Model
from ....regression import DecisionTreeRegressor

class AdaBoostRegressor(Model):
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
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_feats = n_feats

        self.trees = []
        self.alphas = []

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Train the AdaBoost regressor

        Parameters:
            features: Training feature matrix, shape (n_samples, n_features)
            targets: Continuous target vector, shape (n_samples,)
        """
        n_samples = features.shape[0]
        # Initialize uniform sample weights
        sample_weights = np.full(shape=n_samples, fill_value=1.0 / n_samples, dtype=float)

        for _ in range(self.number_of_epochs):
            # Fit a weak learner with current weights
            tree = DecisionTreeRegressor(
                n_feats=self.n_feats,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(features, targets, sample_weights)

            # Predictions and errors
            preds = tree.predict(features)
            errors = np.abs(preds - targets)
            max_error = np.max(errors)
            # If perfect fit, stop early
            if max_error == 0:
                break

            # Normalize errors to [0,1]
            normalized_errors = errors / max_error

            # Compute weighted error
            err_m = np.dot(sample_weights, normalized_errors)
            err_m = np.clip(err_m, 1e-10, 1 - 1e-10)

            # Compute beta and alpha
            beta_m = err_m / (1.0 - err_m)
            alpha_m = self.learn_rate * np.log(1.0 / beta_m)

            # Update sample weights
            sample_weights *= beta_m ** (1.0 - normalized_errors)
            sample_weights /= np.sum(sample_weights)

            # Store the learner and its beta
            self.trees.append(tree)
            self.alphas.append(alpha_m)

    def predict(self,
                test_features: np.ndarray,
                test_targets: np.ndarray = None) -> np.ndarray:
        """
        Predict continuous target values using the trained AdaBoost model.

        Parameters:
            test_features: Test feature matrix, shape (n_samples, n_features)
            test_targets: True target values (optional, for evaluation)

        Returns:
            np.ndarray: Predicted continuous values, shape (n_samples,)
        """

        # Aggregate weighted predictions
        agg = np.zeros(shape=(test_features.shape[0],), dtype=float)
        for tree, alpha in zip(self.trees, self.alphas):
            preds = tree.predict(test_features)
            agg += alpha * preds

        # Final prediction is the weighted average
        total_alpha = np.sum(self.alphas)
        predictions = agg / total_alpha

        if test_targets is not None:
            mse, r2 = self.regression_evaluate(predictions, test_targets)
            print("MSE: {:.5f} R-squared: {:.5f}".format(mse, r2))

        return predictions

    def __str__(self) -> str:
        return "AdaBoost Regressor"
