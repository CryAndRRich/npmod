import numpy as np
from ....regression import DecisionTreeRegressor

class GradientBoostingRegressor():
    def __init__(self,
                 learn_rate: float = 0.01,
                 number_of_epochs: int = 100,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 n_feats: int = None) -> None:
        """
        Gradient Boosting Regressor using custom DecisionTreeRegressor as weak learner

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

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Train the gradient boosting model

        Parameters:
            features: Training feature matrix of shape (n_samples, n_features)
            targets: Training target values
        """
        # Initialize model with the mean of targets
        self.init_pred = np.mean(targets)
        # Current predictions
        current_pred = np.full(shape=targets.shape, fill_value=self.init_pred, dtype=float)
        # Train trees
        for _ in range(self.number_of_epochs):
            # Compute residuals (negative gradient)
            residuals = targets - current_pred
            # Fit a regression tree to residuals
            tree = DecisionTreeRegressor(
                n_feats=self.n_feats,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(features, residuals)

            # Update predictions
            update = tree.predict(features, np.zeros_like(targets), get_accuracy=False)
            current_pred += self.learn_rate * update

            # Store tree
            self.trees.append(tree)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict using the trained gradient boosting model.

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        # Start from initial prediction
        predictions = np.full(shape=(test_features.shape[0],),
                              fill_value=self.init_pred,
                              dtype=float)
        
        # Add contributions from each tree
        for tree in self.trees:
            predictions += self.learn_rate * tree.predict(test_features, np.zeros_like(predictions), get_accuracy=False)

        return predictions
    
    def __str__(self) -> str:
        return "Gradient Boosting Regressor"