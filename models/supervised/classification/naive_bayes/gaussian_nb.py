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
        self.classes = np.unique(targets)
        self.params = {} # mean and variance for each class
        self.class_priors = {} # prior probability for each class
        eps = 1e-9 # to ensure non-zero variance

        for cls in self.classes:
            cls_features = features[targets == cls]
            means = cls_features.mean(axis=0)
            variances = cls_features.var(axis=0) + eps # add epsilon to avoid 0 variance
            self.params[cls] = (means, variances)
            self.class_priors[cls] = cls_features.shape[0] / features.shape[0]

    def _gaussian_log_likelihood(self, 
                                 x: np.ndarray, 
                                 mean: np.ndarray, 
                                 var: np.ndarray) -> float:
        """
        Computes log likelihood under Gaussian distribution
        """
        coeff = -0.5 * np.log(2 * np.pi * var)
        exponent = -((x - mean) ** 2) / (2 * var)
        return np.sum(coeff + exponent)

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
            class_log_posteriors = []
            for cls in self.classes:
                mean, var = self.params[cls]
                prior = np.log(self.class_priors[cls])
                likelihood = self._gaussian_log_likelihood(feature, mean, var)
                class_log_posteriors.append(prior + likelihood)
            predicted_class = self.classes[np.argmax(class_log_posteriors)]
            predictions.append(predicted_class)
        return np.array(predictions)

    def __str__(self) -> str:
        return "Gaussian Naive Bayes"
