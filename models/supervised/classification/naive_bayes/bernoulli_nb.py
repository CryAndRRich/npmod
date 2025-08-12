import numpy as np

class BernoulliNaiveBayes():
    def __init__(self, alpha: int = 1) -> None:
        """
        Initializes the Bernoulli Naive Bayes model

        Parameters:
            alpha: Smoothing parameter for Laplace smoothing
        """
        self.alpha = alpha
        self.eps = 1e-9

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fits the model to the training data by calculating class priors and likelihoods

        Parameters:
            features: Input features for training
            targets: Corresponding targets for the input features
        """
        self.unique_targets = np.unique(targets)
        self.num_classes = len(self.unique_targets)
        self.num_features = features.shape[1]

        self.class_feature_probs = {}
        self.class_priors = {}

        for target in self.unique_targets:
            class_features = features[targets == target]
            mu = (class_features.sum(axis=0) + self.alpha) /  (class_features.shape[0] + 2 * self.alpha)
            self.class_feature_probs[target] = mu
            self.class_priors[target] = np.mean(targets == target)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        num_samples = test_features.shape[0]
        predictions = np.empty(num_samples)

        for i in range(num_samples):
            feature = test_features[i]
            posteriors = []

            for target in self.unique_targets:
                mu = self.class_feature_probs[target]
                prior_log = np.log(self.class_priors[target] + self.eps)

                # Bernoulli log-likelihood
                log_likelihood = np.sum(
                    feature * np.log(mu + self.eps) + 
                    (1 - feature) * np.log(1 - mu + self.eps)
                )
                posteriors.append(prior_log + log_likelihood)

            predictions[i] = self.unique_targets[np.argmax(posteriors)]

        return predictions
    def __str__(self) -> str:
        return "Bernoulli Naive Bayes"
