import numpy as np
from base_model import ModelML

class BernoulliNaiveBayes(ModelML):
    def __init__(self, alpha: int = 1):
        """
        Initializes the Bernoulli Naive Bayes model

        Parameters:
        alpha: Smoothing parameter for Laplace smoothing
        """
        self.alpha = alpha

    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Fits the model to the training data by calculating class priors and likelihoods

        Parameters:
        features: Input features for training
        labels: Corresponding target labels for the input features
        """
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

        _, self.num_features = self.features.shape
        self.num_classes = self.unique_labels.shape[0]

    def bernoulli_distribution(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the Bernoulli distribution for the feature set

        Parameters:
        data: Input feature data for calculating probabilities

        --------------------------------------------------
        Returns:
        mu: Bernoulli probabilities for each feature
        """
        numerator = data.sum(axis=0) + self.alpha
        denominator = data.shape[0] + 2 * self.alpha

        mu = numerator / denominator
        return mu
    
    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray) -> None:
        """
        Predicts the class labels for test data using the trained Bernoulli Naive Bayes model

        Parameters:
        test_features: Input features for testing
        test_labels: Corresponding target labels for the test features
        """
        num_samples, _ = test_features.shape

        predictions = np.empty(num_samples)
        for ind, feature in enumerate(test_features):
            posteriors = []
            for _, label in enumerate(self.unique_labels):
                prior = np.log((self.labels == label).mean())
                label_feature = self.features[self.labels == label, :]

                mu = self.bernoulli_distribution(label_feature)
                likelihood = np.log(mu).dot(feature.T) + np.log(1 - mu).dot(1 - feature.T)

                posteriors.append(prior + likelihood)
            
            predictions[ind] = self.unique_labels[np.argmax(posteriors)]

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Alpha: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.alpha, accuracy, f1))
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Bernoulli Naive Bayes model
        """
        return "Bernoulli Naive Bayes"
