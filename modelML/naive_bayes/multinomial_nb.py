import numpy as np
from ..base_model import ModelML

class MultinomialNaiveBayes(ModelML):
    def __init__(self, alpha: int = 1) -> None:
        """
        Initializes the Multinomial Naive Bayes model.

        --------------------------------------------------
        Parameters:
            alpha: Smoothing parameter for Laplace smoothing 
        """
        self.alpha = alpha

    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Fits the Multinomial Naive Bayes model by calculating feature counts and total counts for each class.

        --------------------------------------------------
        Parameters:
            features: Input features for training
            labels: Corresponding target labels for the input features
        """
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

        _, self.num_features = self.features.shape
        self.num_classes = self.unique_labels.shape[0]

        # N_yi stores the sum of feature counts for each feature in each class
        self.N_yi = np.zeros((self.num_classes, self.num_features))
        # N_y stores the total feature count for each class
        self.N_y = np.zeros((self.num_classes))

        # Calculate feature counts for each class
        for label in self.unique_labels:
            indices = np.argwhere(self.labels == label).flatten()  # Indices of samples for the current class
            columnwise_sum = []
            for j in range(self.num_features):
                columnwise_sum.append(np.sum(self.features[indices, j]))  # Sum the feature counts for the class
                
            self.N_yi[label] = columnwise_sum  # Store the feature counts
            self.N_y[label] = np.sum(columnwise_sum)  # Store the total count of features for the class

    def multinomial_distribution(self, 
                                 data: np.ndarray, 
                                 class_idx: int) -> float:
        """
        Computes the probability of the input data using the multinomial distribution for a given class.

        --------------------------------------------------
        Parameters:
            data: Input feature values
            class_idx: The class index for which to calculate the multinomial probability

        --------------------------------------------------
        Returns:
            prob: The probability of the input data for the given class
        """
        temp = []
        m = data.shape[0]

        # Calculate the probability for each feature and multiply them together
        for i in range(m):
            numerator = self.N_yi[class_idx, i] + self.alpha  # Apply Laplace smoothing to the feature count
            denominator = self.N_y[class_idx] + (self.alpha * self.num_features)  # Apply smoothing to the total count
            
            theta = (numerator / denominator) ** data[i]  # Multinomial likelihood for the feature
            temp.append(theta)

        prob = np.prod(temp)
        return prob  # Return the product of the probabilities

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
            joint_likelihood = []

            # Calculate joint likelihood for each class
            for label_idx, label in enumerate(self.unique_labels):
                prior = np.log((self.labels == label).mean())  # Calculate prior log-probability of the class
                likelihood = np.log(self.multinomial_distribution(feature, label_idx))  # Calculate log-likelihood
                
                joint_likelihood.append(prior + likelihood)  # Add prior and likelihood to get joint likelihood

            denominator = np.sum(joint_likelihood)  # Normalization factor
            for likelihood in joint_likelihood:
                posteriors.append(likelihood - denominator)  # Normalize the posterior probability

            # Choose the class with the highest posterior probability
            predictions[ind] = self.unique_labels[np.argmax(posteriors)]

        if get_accuracy:
            # Evaluate accuracy and F1-score
            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Alpha: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.alpha, accuracy, f1))

        return predictions

    def __str__(self) -> str:
        """
        Returns a string representation of the Multinomial Naive Bayes model.
        """
        return "Multinomial Naive Bayes"
