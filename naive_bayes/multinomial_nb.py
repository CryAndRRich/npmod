import numpy as np
from base_model import ModelML

class MultinomialNaiveBayes(ModelML):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.unique_labels = np.unique(labels)

        _, self.num_features = self.features.shape
        self.num_classes = self.unique_labels.shape[0]

        self.N_yi = np.zeros((self.num_classes, self.num_features))
        self.N_y = np.zeros((self.num_classes))

        for label in self.unique_labels:
            indices = np.argwhere(self.labels == label).flatten()
            columnwise_sum = []
            for j in range(self.num_features):
                columnwise_sum.append(np.sum(self.features[indices,j]))
                
            self.N_yi[label] = columnwise_sum 
            self.N_y[label] = np.sum(columnwise_sum)
    
    def multinomial_distribution(self, data, h):
        temp = []
        m = data.shape[0]
        for i in range(m):
            numerator = self.N_yi[h, i] + self.alpha
            denominator = self.N_y[h] + (self.alpha * self.num_features)
            
            theta = (numerator / denominator) ** data[i]
            temp.append(theta)
        
        return np.prod(temp)
    
    def predict(self, test_features, test_labels):
        num_samples, _ = test_features.shape

        predictions = np.empty(num_samples)
        for ind, feature in enumerate(test_features):
            posteriors = []
            joint_likelihood = []
            for label_idx, label in enumerate(self.unique_labels):
                prior = np.log((self.labels == label).mean())
                likelihood = np.log(self.multinomial_distribution(feature, label_idx))
                
                joint_likelihood.append(prior + likelihood)

            denominator = np.sum(joint_likelihood)
            for likelihood in joint_likelihood:
                posteriors.append(likelihood - denominator)

            predictions[ind] = self.unique_labels[np.argmax(posteriors)]

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Alpha: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.alpha, accuracy, f1))

    def __str__(self):
        return "Multinomial Naive Bayes"