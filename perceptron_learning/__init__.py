from .perceptron_numpy import PerceptronLearningNumpy
from .perceptron_pytorch import PerceptronLearningPytorch

class PerceptronLearning():
    def __init__(self, learn_rate, number_of_epochs, type="numpy"):
        if type == "numpy":
            self.inherit = PerceptronLearningNumpy(learn_rate, number_of_epochs)
        elif type == "pytorch":
            self.inherit = PerceptronLearningPytorch(learn_rate, number_of_epochs)
        else: 
            raise ValueError(f"Type must be 'numpy' or 'pytorch'")
    
    def fit(self, features, labels):
        self.inherit.fit(features, labels)

    def predict(self, test_features, test_labels):
        self.inherit.predict(test_features, test_labels)
    
    def __str__(self):
        return self.inherit.__str__()