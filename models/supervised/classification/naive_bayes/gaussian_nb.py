import numpy as np

class GaussianNaiveBayes():
    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fits the Gaussian Naive Bayes model by calculating the mean and variance 
        for each feature in each class.

        Parameters:
            features: Input features for training
            targets: Corresponding targets for the input features
        """
        self.features = features
        self.targets = targets
        self.unique_targets = np.unique(targets)

        # Calculate mean and variance for each feature in each class
        self.params = []
        for target in self.unique_targets:
            target_features = self.features[self.targets == target]  # Select features of the current class
            # Store the mean and variance for each feature of the class
            self.params.append([(col.mean(), col.var()) for col in target_features.T])

    def gaussian_distribution(self, 
                              data: np.ndarray, 
                              sigma: float, 
                              mu: float) -> np.ndarray:
        """
        Computes the Gaussian distribution for the input data using the given mean (mu) and variance (sigma).

        Parameters:
            data: Input feature values
            sigma: Variance of the Gaussian distribution
            mu: Mean of the Gaussian distribution

        Returns:
            prob: Gaussian probability values for the input data
        """
        eps = 1e-15  # Small value to avoid division by zero

        # Compute the Gaussian coefficient and exponent
        coeff = 1 / np.sqrt(2 * np.pi * sigma + eps)
        exponent = np.exp(-((data - mu) ** 2 / (2 * sigma + eps)))

        # Return the Gaussian probability
        prob = coeff * exponent + eps
        return prob

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
            for target_idx, target in enumerate(self.unique_targets):
                prior = np.log((self.targets == target).mean())  # Calculate prior log-probability of the class

                # Combine the feature with the mean and variance
                pairs = zip(feature, self.params[target_idx])
                # Calculate the log-likelihood of the data given the Gaussian distribution for each feature
                likelihood = np.sum([np.log(self.gaussian_distribution(f, m, v)) for f, (m, v) in pairs])

                posteriors.append(prior + likelihood)  # Add prior and likelihood to get the posterior

            # Choose the class with the highest posterior probability
            predictions[ind] = self.unique_targets[np.argmax(posteriors)]

        return predictions

    def __str__(self) -> str:
        return "Gaussian Naive Bayes"
