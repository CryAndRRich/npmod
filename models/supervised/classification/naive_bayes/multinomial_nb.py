import numpy as np

class MultinomialNaiveBayes():
    def __init__(self, alpha: int = 1) -> None:
        """
        Initializes the Multinomial Naive Bayes model.

        Parameters:
            alpha: Smoothing parameter for Laplace smoothing 
        """
        self.alpha = alpha

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fits the Multinomial Naive Bayes model by calculating feature counts and total counts for each class.

        Parameters:
            features: Input features for training
            targets: Corresponding targets for the input features
        """
        self.classes = np.unique(targets)
        self.class_indices = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.num_features = features.shape[1]

        # Count of feature values per class
        self.feature_counts = np.zeros((self.num_classes, self.num_features))
        # Total count of all features per class
        self.class_totals = np.zeros(self.num_classes)
        # Class prior probabilities
        self.class_priors = np.zeros(self.num_classes)

        for cls in self.classes:
            idx = self.class_indices[cls]
            X_cls = features[targets == cls]
            self.feature_counts[idx] = X_cls.sum(axis=0)
            self.class_totals[idx] = self.feature_counts[idx].sum()
            self.class_priors[idx] = X_cls.shape[0] / features.shape[0]

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        predictions = []
        for feature in test_features:
            log_probs = []
            for cls in self.classes:
                idx = self.class_indices[cls]
                log_prior = np.log(self.class_priors[idx])

                # Apply Laplace smoothing
                numerator = self.feature_counts[idx] + self.alpha
                denominator = self.class_totals[idx] + self.alpha * self.num_features

                log_likelihood = np.sum(feature * np.log(numerator / denominator))
                log_probs.append(log_prior + log_likelihood)

            predicted_class = self.classes[np.argmax(log_probs)]
            predictions.append(predicted_class)
        return np.array(predictions)

    def __str__(self) -> str:
        return "Multinomial Naive Bayes"
