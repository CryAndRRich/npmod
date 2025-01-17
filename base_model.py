from sklearn.metrics import accuracy_score, f1_score

class ModelML():
    def __init__(self):
        pass
    
    def fit(self, features, labels):
        pass

    def predict(self, test_features, test_labels):
        pass

    def evaluate(self, predictions, test_labels):
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average="weighted", zero_division=0)

        return accuracy, f1
    
    def __str__(self):
        pass