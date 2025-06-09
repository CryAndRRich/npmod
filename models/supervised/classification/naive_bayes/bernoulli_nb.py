import numpy as np

class BernoulliNaiveBayes():
    def __init__(self, alpha: int = 1) -> None:
        """
        Initializes the Bernoulli Naive Bayes model

        Parameters:
            alpha: Smoothing parameter for Laplace smoothing
        """
        self.alpha = alpha

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fits the model to the training data by calculating class priors and likelihoods

        Parameters:
            features: Input features for training
            targets: Corresponding targets for the input features
        """
        self.features = features
        self.targets = targets
        self.unique_targets = np.unique(targets)

        _, self.num_features = self.features.shape
        self.num_classes = self.unique_targets.shape[0]

    def bernoulli_distribution(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the Bernoulli distribution for the feature set

        Parameters:
            data: Input feature data for calculating probabilities

        Returns:
            mu: Bernoulli probabilities for each feature
        """
        numerator = data.sum(axis=0) + self.alpha
        denominator = data.shape[0] + 2 * self.alpha

        mu = numerator / denominator
        return mu
    
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
            for _, target in enumerate(self.unique_targets):
                prior = np.log((self.targets == target).mean())
                target_feature = self.features[self.targets == target, :]

                mu = self.bernoulli_distribution(target_feature)
                likelihood = np.log(mu).dot(feature.T) + np.log(1 - mu).dot(1 - feature.T)

                posteriors.append(prior + likelihood)
            
            predictions[ind] = self.unique_targets[np.argmax(posteriors)]

        return predictions
    
    def __str__(self) -> str:
        return "Bernoulli Naive Bayes"
