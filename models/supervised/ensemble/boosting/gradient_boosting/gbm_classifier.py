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

        self.classes_ = None
        self.trees = []

    def _softmax(self, features: np.ndarray) -> np.ndarray:
        exp_X = np.exp(features - np.max(features, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Train the gradient boosting classifier

        Parameters:
            features: Training feature matrix
            targets: Training target values
        """
        n_samples, _ = features.shape
        self.classes_ = np.unique(targets)
        n_classes = len(self.classes_)

        # Initialize raw scores F_0 = 0 for all classes
        raw_scores = np.zeros((n_samples, n_classes))

        self.trees = []

        for _ in range(self.number_of_epochs):
            trees_m = []
            prob = self._softmax(raw_scores)

            for k, cls in enumerate(self.classes_):
                # Compute pseudo-residuals for class k
                y_k = (targets == cls).astype(float)
                residuals = y_k - prob[:, k]

                # Fit regression tree to residuals
                tree = DecisionTreeRegressor(
                    n_feats=self.n_feats,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )
                tree.fit(features, residuals)
                trees_m.append(tree)

                # Update raw_scores for class k
                raw_scores[:, k] += self.learn_rate * tree.predict(features)

            self.trees.append(trees_m)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict using the trained gradient boosting model

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        n_samples = test_features.shape[0]
        n_classes = len(self.classes_)
        raw_scores = np.zeros((n_samples, n_classes))

        for trees_m in self.trees:
            for k, tree in enumerate(trees_m):
                raw_scores[:, k] += self.learn_rate * tree.predict(test_features)

        prob = self._softmax(raw_scores)
        return self.classes_[np.argmax(prob, axis=1)]
    
    def __str__(self) -> str:
        return "Gradient Boosting Classifier"