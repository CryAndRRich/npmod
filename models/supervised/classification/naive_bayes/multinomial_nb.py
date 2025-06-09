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
        self.features = features
        self.targets = targets
        self.unique_targets = np.unique(targets)

        _, self.num_features = self.features.shape
        self.num_classes = self.unique_targets.shape[0]

        # N_yi stores the sum of feature counts for each feature in each class
        self.N_yi = np.zeros((self.num_classes, self.num_features))
        # N_y stores the total feature count for each class
        self.N_y = np.zeros((self.num_classes))

        # Calculate feature counts for each class
        for target in self.unique_targets:
            indices = np.argwhere(self.targets == target).flatten()  # Indices of samples for the current class
            columnwise_sum = []
            for j in range(self.num_features):
                columnwise_sum.append(np.sum(self.features[indices, j]))  # Sum the feature counts for the class
                
            self.N_yi[target] = columnwise_sum  # Store the feature counts
            self.N_y[target] = np.sum(columnwise_sum)  # Store the total count of features for the class

    def multinomial_distribution(self, 
                                 data: np.ndarray, 
                                 class_idx: int) -> float:
        """
        Computes the probability of the input data using the multinomial distribution for a given class.

        Parameters:
            data: Input feature values
            class_idx: The class index for which to calculate the multinomial probability

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
            joint_likelihood = []

            # Calculate joint likelihood for each class
            for target_idx, target in enumerate(self.unique_targets):
                prior = np.log((self.targets == target).mean())  # Calculate prior log-probability of the class
                likelihood = np.log(self.multinomial_distribution(feature, target_idx))  # Calculate log-likelihood
                
                joint_likelihood.append(prior + likelihood)  # Add prior and likelihood to get joint likelihood

            denominator = np.sum(joint_likelihood)  # Normalization factor
            for likelihood in joint_likelihood:
                posteriors.append(likelihood - denominator)  # Normalize the posterior probability

            # Choose the class with the highest posterior probability
            predictions[ind] = self.unique_targets[np.argmax(posteriors)]

        return predictions

    def __str__(self) -> str:
        return "Multinomial Naive Bayes"
