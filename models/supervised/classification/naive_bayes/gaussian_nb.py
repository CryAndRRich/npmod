import numpy as np
from ..base import Model

class GaussianNaiveBayes(Model):
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Fits the Gaussian Naive Bayes model by calculating the mean and variance 
        for each feature in each class.

        --------------------------------------------------
        Parameters:
            features: Input features for training
            labels: Corresponding target labels for the input features
        """
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

        # Calculate mean and variance for each feature in each class
        self.params = []
        for label in self.unique_labels:
            label_features = self.features[self.labels == label]  # Select features of the current class
            # Store the mean and variance for each feature of the class
            self.params.append([(col.mean(), col.var()) for col in label_features.T])

    def gaussian_distribution(self, 
                              data: np.ndarray, 
                              sigma: float, 
                              mu: float) -> np.ndarray:
        """
        Computes the Gaussian distribution for the input data using the given mean (mu) and variance (sigma).

        --------------------------------------------------
        Parameters:
            data: Input feature values
            sigma: Variance of the Gaussian distribution
            mu: Mean of the Gaussian distribution

        --------------------------------------------------
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

    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray,
                get_accuracy: bool = True) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        --------------------------------------------------
        Parameters:
            test_features: The input features for testing
            test_labels: The true target labels corresponding to the test features
            get_accuracy: If True, calculates and prints the accuracy of predictions

        --------------------------------------------------
        Returns:
            predictions: The prediction labels
        """
        num_samples, _ = test_features.shape

        predictions = np.empty(num_samples)
        for ind, feature in enumerate(test_features):
            posteriors = []
            # Calculate posterior probabilities for each class
            for label_idx, label in enumerate(self.unique_labels):
                prior = np.log((self.labels == label).mean())  # Calculate prior log-probability of the class

                # Combine the feature with the mean and variance
                pairs = zip(feature, self.params[label_idx])
                # Calculate the log-likelihood of the data given the Gaussian distribution for each feature
                likelihood = np.sum([np.log(self.gaussian_distribution(f, m, v)) for f, (m, v) in pairs])

                posteriors.append(prior + likelihood)  # Add prior and likelihood to get the posterior

            # Choose the class with the highest posterior probability
            predictions[ind] = self.unique_labels[np.argmax(posteriors)]

        if get_accuracy:
            # Evaluate accuracy and F1-score
            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Accuracy: {:.5f} F1-score: {:.5f}".format(accuracy, f1))
        
        return predictions

    def __str__(self) -> str:
        """
        Returns a string representation of the Gaussian Naive Bayes model
        """
        return "Gaussian Naive Bayes"
