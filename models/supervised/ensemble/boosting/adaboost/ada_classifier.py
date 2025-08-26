import numpy as np
from ....classification import DecisionTreeClassifier

class AdaBoostClassifier():
    def __init__(self,
                 learn_rate: float = 1.0,
                 number_of_epochs: int = 50,
                 max_depth: int = 1,
                 min_samples_split: int = 2,
                 n_feats: int = None) -> None:
        """
        AdaBoost Classifier using custom DecisionTreeRegressor as weak learner

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
        self.classes_ = None

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Train the AdaBoost classifier

        Parameters:
            features: Training feature matrix
            targets: Target vector
        """
        n_samples = features.shape[0]
        self.classes_ = np.unique(targets)
        K = len(self.classes_)

        # initialize uniform weights
        sample_weights = np.full(n_samples, 1.0 / n_samples)

        for _ in range(self.number_of_epochs):
            # weak learner
            stump = DecisionTreeClassifier(
                algorithm = "ID3"
            )
            stump.fit(features, targets)

            # predictions
            preds = stump.predict(features)

            # weighted error
            incorrect = (preds != targets).astype(float)
            err = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # alpha (SAMME)
            alpha = self.learn_rate * (np.log((1 - err) / err) + np.log(K - 1))

            # update weights
            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)

            # store
            self.trees.append(stump)
            self.alphas.append(alpha)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict using the trained AdaBoost model

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted labels
        """
        n_samples = test_features.shape[0]
        K = len(self.classes_)
        class_scores = np.zeros((n_samples, K))

        for stump, alpha in zip(self.trees, self.alphas):
            preds = stump.predict(test_features)
            for i, c in enumerate(self.classes_):
                class_scores[:, i] += alpha * (preds == c)

        return self.classes_[np.argmax(class_scores, axis=1)]

    def __str__(self) -> str:
        return "AdaBoost Classifier"
