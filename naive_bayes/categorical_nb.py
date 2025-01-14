import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class CategoricalNaiveBayes():
    def __init__(self, features, labels, alpha=1):
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.alpha = alpha

        self.num_samples, self.num_features = self.features.shape
        self.num_classes = self.unique_labels.shape[0]

        self.class_priors = np.zeros(self.num_classes)
        for idx, label in enumerate(self.unique_labels):
            self.class_priors[idx] = np.sum(self.labels == label) / self.num_samples

        self.conditional_probs = []
        for feature_idx in range(self.num_features):
            unique_values = np.unique(self.features[:, feature_idx])
            cond_prob = np.zeros((self.num_classes, len(unique_values)))

            for class_idx, label in enumerate(self.unique_labels):
                indices = np.argwhere(self.labels == label).flatten()
                feature_values = self.features[indices, feature_idx]

                for value_idx, value in enumerate(unique_values):
                    cond_prob[class_idx, value_idx] = (
                        np.sum(feature_values == value) + self.alpha
                        ) / (len(feature_values) + self.alpha * len(unique_values))

            self.conditional_probs.append((unique_values, cond_prob))

    def categorical_distribution(self, data, class_idx):
        log_prob = np.log(self.class_priors[class_idx])

        for feature_idx in range(self.num_features):
            unique_values, cond_prob = self.conditional_probs[feature_idx]
            value_idx = np.where(unique_values == data[feature_idx])[0]
            if len(value_idx) > 0:
                log_prob += np.log(cond_prob[class_idx, value_idx[0]])
            else:
                log_prob += np.log(self.alpha / (self.alpha * len(unique_values)))

        return log_prob

    def train_model(self, test_features, test_labels):
        num_samples, _ = test_features.shape

        predictions = np.empty(num_samples)
        for ind, feature in enumerate(test_features):
            posteriors = []
            for class_idx in range(self.num_classes):
                likelihood = self.categorical_distribution(feature, class_idx)
                posteriors.append(likelihood)

            predictions[ind] = self.unique_labels[np.argmax(posteriors)]

        accuracy, f1 = self.test_model(predictions, test_labels)
        print("CNB model Alpha: {} Accuracy: {:.3f}% F1-score: {:.3f}".format(self.alpha, accuracy, f1))

    def test_model(self, predictions, test_labels):
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average="weighted", zero_division=0)

        return accuracy * 100, f1