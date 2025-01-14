import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class BernoulliNaiveBayes():
    def __init__(self, features, labels, alpha=1):
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.alpha = alpha

        _, self.num_features = self.features.shape
        self.num_classes = self.unique_labels.shape[0]
    
    def bernoulli_distribution(self, data):
        numerator = data.sum(axis=0) + self.alpha
        denominator = data.shape[0] + 2 * self.alpha

        mu = numerator / denominator
        return mu
    
    def train_model(self, test_features, test_labels):
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

        accuracy, f1 = self.test_model(predictions, test_labels)
        print("BNB model   Alpha: {} Accuracy: {:.3f}% F1-score: {:.3f}".format(self.alpha, accuracy, f1))
    
    def test_model(self, predictions, test_labels):
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average="weighted", zero_division=0)

        return accuracy * 100, f1
    