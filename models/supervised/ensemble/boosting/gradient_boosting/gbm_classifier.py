import numpy as np
from ....regression import DecisionTreeRegressor

class GradientBoostingClassifier():
    def __init__(self,
                 learn_rate: float = 0.01,
                 number_of_epochs: int = 100,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 n_feats: int = None) -> None:
        """
        Gradient Boosting Classifier using custom DecisionTreeRegressor as base learner
        for binary classification with logistic loss

        Parameters:
            learn_rate: The learning rate for the gradient descent
            number_of_epochs: The number of training iterations to run
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split an internal node
            n_feats: Number of features to consider when searching for the best split
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_feats = n_feats

        self.init_pred = None
        self.trees = []

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function

        Parameters:
            x: Input array

        Returns:
            np.ndarray: Sigmoid of input
        """
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Train the gradient boosting classifier

        Parameters:
            features: Training feature matrix of shape (n_samples, n_features)
            targets: Training target values
        """
        # Initialize raw prediction with log-odds of positive class
        positive_rate = np.clip(np.mean(targets), 1e-6, 1 - 1e-6)
        self.init_pred = np.log(positive_rate / (1 - positive_rate))
        raw_pred = np.full_like(targets, fill_value=self.init_pred, dtype=float)

        # Boosting iterations
        for _ in range(self.number_of_epochs):
            # Compute pseudo-residuals (negative gradient of logistic loss)
            prob = self._sigmoid(raw_pred)
            residuals = targets - prob

            # Fit regression tree to residuals
            tree = DecisionTreeRegressor(
                n_feats=self.n_feats,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(features, residuals)

            # Update raw predictions
            update = tree.predict(features, np.zeros_like(residuals), get_accuracy=False)
            raw_pred += self.learn_rate * update

            # Store the fitted tree
            self.trees.append(tree)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict using the trained gradient boosting model.

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        # Initialize raw scores with the initial log-odds
        raw_pred = np.full(shape=(test_features.shape[0],),
                           fill_value=self.init_pred,
                           dtype=float)

        # Add contributions from each regression tree
        for tree in self.trees:
            update = tree.predict(test_features, np.zeros_like(raw_pred), get_accuracy=False)
            raw_pred += self.learn_rate * update

        # Convert raw scores to probabilities and then to binary labels
        prob = self._sigmoid(raw_pred)
        predictions = (prob >= 0.5).astype(int)

        return predictions
    
    def __str__(self) -> str:
        return "Gradient Boosting Classifier"