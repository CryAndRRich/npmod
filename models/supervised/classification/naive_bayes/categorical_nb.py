import numpy as np

class CategoricalNaiveBayes():
    def __init__(self, alpha: int = 1) -> None:
        """
        Initializes the Categorical Naive Bayes model

        Parameters:
            alpha: Smoothing parameter for Laplace smoothing 
        """
        self.alpha = alpha

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fits the model to the training data by calculating class priors and conditional probabilities

        Parameters:
            features: Input features for training
            targets: Corresponding targets for the input features
        """
        self.features = features
        self.targets = targets
        self.unique_targets = np.unique(targets)

        self.num_samples, self.num_features = self.features.shape
        self.num_classes = self.unique_targets.shape[0]

        # Compute class priors based on target frequency
        self.class_priors = np.zeros(self.num_classes)
        for idx, target in enumerate(self.unique_targets):
            self.class_priors[idx] = np.sum(self.targets == target) / self.num_samples

        # Compute conditional probabilities for each feature and class
        self.conditional_probs = []
        for feature_idx in range(self.num_features):
            unique_values = np.unique(self.features[:, feature_idx])  # Find unique values in each feature
            cond_prob = np.zeros((self.num_classes, len(unique_values)))

            for class_idx, target in enumerate(self.unique_targets):
                indices = np.argwhere(self.targets == target).flatten()  # Find indices of each class
                feature_values = self.features[indices, feature_idx]   # Extract feature values for the class

                # Compute conditional probabilities with Laplace smoothing
                for value_idx, value in enumerate(unique_values):
                    cond_prob[class_idx, value_idx] = (
                        np.sum(feature_values == value) + self.alpha
                        ) / (len(feature_values) + self.alpha * len(unique_values))

            self.conditional_probs.append((unique_values, cond_prob))  # Store the conditional probabilities

    def categorical_distribution(self, 
                                 data: np.ndarray, 
                                 class_idx: int) -> float:
        """
        Computes the log probability of the data given a class using the categorical distribution

        Parameters:
            data: Input feature values
            class_idx: The class index for which to calculate the log-probability

        Returns:
            log_prob: The log probability of the input data for the given class
        """
        # Start with the class prior (log form)
        log_prob = np.log(self.class_priors[class_idx])

        # Add the log-probabilities for each feature based on the conditional probabilities
        for feature_idx in range(self.num_features):
            unique_values, cond_prob = self.conditional_probs[feature_idx]
            value_idx = np.where(unique_values == data[feature_idx])[0]
            
            # If the feature value exists, use the corresponding conditional probability
            if len(value_idx) > 0:
                log_prob += np.log(cond_prob[class_idx, value_idx[0]])
            # If feature value not found, apply Laplace smoothing
            else:
                log_prob += np.log(self.alpha / (self.alpha * len(unique_values)))

        return log_prob

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        num_samples, _ = test_features.shape

        predictions = np.empty(num_samples)
        for ind, feature in enumerate(test_features):
            posteriors = []
            # Calculate posterior probabilities for each class
            for class_idx in range(self.num_classes):
                likelihood = self.categorical_distribution(feature, class_idx)
                posteriors.append(likelihood)

            # Choose the class with the highest posterior probability
            predictions[ind] = self.unique_targets[np.argmax(posteriors)]

        return predictions

    def __str__(self) -> str:
        return "Categorical Naive Bayes"
