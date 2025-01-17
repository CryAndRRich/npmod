import numpy as np
from base_model import ModelML

class CategoricalNaiveBayes(ModelML):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

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

    def predict(self, test_features, test_labels):
        num_samples, _ = test_features.shape

        predictions = np.empty(num_samples)
        for ind, feature in enumerate(test_features):
            posteriors = []
            for class_idx in range(self.num_classes):
                likelihood = self.categorical_distribution(feature, class_idx)
                posteriors.append(likelihood)

            predictions[ind] = self.unique_labels[np.argmax(posteriors)]

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Alpha: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.alpha, accuracy, f1))

    def __str__(self):
        return "Categorical Naive Bayes"