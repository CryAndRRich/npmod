import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class GaussianNaiveBayes():
    def __init__(self, features, labels):
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

    def train_model(self, test_features, test_labels):
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

        accuracy, f1 = self.test_model(predictions, test_labels)
        print("GNB model   Accuracy: {:.3f}% F1-score: {:.3f}".format(accuracy, f1))
    
    def test_model(self, predictions, test_labels):
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average="weighted", zero_division=0)

        return accuracy * 100, f1
    