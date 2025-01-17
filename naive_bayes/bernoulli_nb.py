import numpy as np
from base_model import ModelML

class BernoulliNaiveBayes(ModelML):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

        _, self.num_features = self.features.shape
        self.num_classes = self.unique_labels.shape[0]

    def bernoulli_distribution(self, data):
        numerator = data.sum(axis=0) + self.alpha
        denominator = data.shape[0] + 2 * self.alpha

        mu = numerator / denominator
        return mu
    
    def predict(self, test_features, test_labels):
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
    
    def __str__(self):
        return "Bernoulli Naive Bayes"
    