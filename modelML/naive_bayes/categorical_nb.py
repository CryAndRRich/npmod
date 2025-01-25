import numpy as np
from ..base_model import ModelML

class CategoricalNaiveBayes(ModelML):
    def __init__(self, alpha: int = 1):
        """
        Initializes the Categorical Naive Bayes model

        Parameters:
        alpha: Smoothing parameter for Laplace smoothing 
        """
        self.alpha = alpha

    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Fits the model to the training data by calculating class priors and conditional probabilities

        Parameters:
        features: Input features for training
        labels: Corresponding target labels for the input features
        """
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

        self.num_samples, self.num_features = self.features.shape
        self.num_classes = self.unique_labels.shape[0]

        # Compute class priors based on label frequency
        self.class_priors = np.zeros(self.num_classes)
        for idx, label in enumerate(self.unique_labels):
            self.class_priors[idx] = np.sum(self.labels == label) / self.num_samples

        # Compute conditional probabilities for each feature and class
        self.conditional_probs = []
        for feature_idx in range(self.num_features):
            unique_values = np.unique(self.features[:, feature_idx])  # Find unique values in each feature
            cond_prob = np.zeros((self.num_classes, len(unique_values)))

            for class_idx, label in enumerate(self.unique_labels):
                indices = np.argwhere(self.labels == label).flatten()  # Find indices of each class
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

        --------------------------------------------------
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

    def predict(self, test_features: np.ndarray, test_labels: np.ndarray) -> None:
        """
        Predicts the class labels for test data using the trained Categorical Naive Bayes model

        Parameters:
        test_features: Input features for testing
        test_labels: Corresponding target labels for the test features
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
            predictions[ind] = self.unique_labels[np.argmax(posteriors)]

        # Evaluate accuracy and F1-score
        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Alpha: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.alpha, accuracy, f1))

    def __str__(self) -> str:
        """
        Returns a string representation of the Categorical Naive Bayes model
        """
        return "Categorical Naive Bayes"
