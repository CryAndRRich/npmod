import numpy as np
from .....base import Model
from ....regression import DecisionTreeRegressor

class AdaBoostClassifier(Model):
    def __init__(self,
                 learn_rate: float = 1.0,
                 number_of_epochs: int = 50,
                 max_depth: int = 1,
                 min_samples_split: int = 2,
                 n_feats: int = None) -> None:
        """
        AdaBoost Classifier using custom DecisionTreeRegressor as weak learner
        for binary classification with exponential loss

        Parameters:
            learn_rate: The learning rate (shrinkage) for the weak learner weights
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
        Train the AdaBoost classifier

        Parameters:
            features: Training feature matrix, shape (n_samples, n_features)
            targets: Binary target vector of 0s and 1s
        """
        n_samples = features.shape[0]
        # Convert targets to {-1, +1}
        y = np.where(targets == 1, 1, -1)

        # Initialize sample weights uniformly
        sample_weights = np.full(shape=n_samples, fill_value=1.0 / n_samples, dtype=float)

        for _ in range(self.number_of_epochs):
            # Fit a regression tree to the weighted samples
            tree = DecisionTreeRegressor(
                n_feats=self.n_feats,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            # Assuming DecisionTreeRegressor supports sample_weight kwarg
            tree.fit(features, y, sample_weight=sample_weights)  

            # Get predictions in {-1, +1}
            pred_raw = tree.predict(features, np.zeros_like(y), get_accuracy=False)
            h = np.where(pred_raw >= 0.5, 1, -1)

            # Compute weighted error
            incorrect = (h != y).astype(float)
            err = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            # Avoid division by zero / perfect fit
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # Compute model weight
            alpha = self.learn_rate * 0.5 * np.log((1 - err) / err)

            # Update sample weights
            sample_weights *= np.exp(-alpha * y * h)
            sample_weights /= np.sum(sample_weights)

            # Save the weak learner and its weight
            self.trees.append(tree)
            self.alphas.append(alpha)

    def predict(self,
                test_features: np.ndarray,
                test_targets: np.ndarray = None) -> np.ndarray:
        """
        Predict using the trained AdaBoost model.

        Parameters:
            test_features: Test feature matrix, shape (n_samples, n_features)
            test_targets: True binary labels (optional, for evaluation)

        Returns:
            np.ndarray: Predicted binary labels (0 or 1)
        """
        # Aggregate weighted predictions
        agg = np.zeros(shape=(test_features.shape[0],), dtype=float)
        for tree, alpha in zip(self.trees, self.alphas):
            pred_raw = tree.predict(test_features, np.zeros_like(agg), get_accuracy=False)
            h = np.where(pred_raw >= 0.5, 1, -1)
            agg += alpha * h

        # Convert sign to (0, 1)
        pred_sign = np.sign(agg)
        predictions = np.where(pred_sign > 0, 1, 0)

        if test_targets is not None:
            accuracy, f1 = self.classification_evaluate(predictions, test_targets)
            print("Accuracy: {:.5f} F1-score: {:.5f}".format(accuracy, f1))

        return predictions

    def __str__(self) -> str:
        return "AdaBoost Classifier"
