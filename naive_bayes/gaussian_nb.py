import numpy as np
from base_model import ModelML

class GaussianNaiveBayes(ModelML):
    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

        self.params = []
        for label in self.unique_labels:
            label_features = self.features[self.labels == label]
            self.params.append([(col.mean(), col.var()) for col in label_features.T])
    
    def gaussian_distribution(self, data, sigma, mu):
        eps = 1e-15

        coeff = 1 / np.sqrt(2 * np.pi * sigma + eps)
        exponent = np.exp(-((data - mu) ** 2 / (2 * sigma + eps)))

        return coeff * exponent + eps

    def predict(self, test_features, test_labels):
        num_samples, _ = test_features.shape

        predictions = np.empty(num_samples)
        for ind, feature in enumerate(test_features):
            posteriors = []
            for label_idx, label in enumerate(self.unique_labels):
                prior = np.log((self.labels == label).mean())

                pairs = zip(feature, self.params[label_idx])
                likelihood = np.sum([np.log(self.gaussian_distribution(f, m, v)) for f, (m, v) in pairs])

                posteriors.append(prior + likelihood)

            predictions[ind] = self.unique_labels[np.argmax(posteriors)]

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Accuracy: {:.5f} F1-score: {:.5f}".format(accuracy, f1))
    
    def __str__(self):
        return "Gaussian Naive Bayes"