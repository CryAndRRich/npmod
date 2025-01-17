from .softmax_regression_numpy import SoftmaxRegressionNumpy
from .softmax_regression_pytorch import SoftmaxRegressionPytorch

class SoftmaxRegression():
    def __init__(self, learn_rate, number_of_epochs, number_of_classes=2, type="numpy"):
        if type == "numpy":
            self.inherit = SoftmaxRegressionNumpy(learn_rate, number_of_epochs, number_of_classes)
        elif type == "pytorch":
            self.inherit = SoftmaxRegressionPytorch(learn_rate, number_of_epochs, number_of_classes)
        else: 
            raise ValueError(f"Type must be 'numpy' or 'pytorch'")
    
    def fit(self, features, labels):
        self.inherit.fit(features, labels)

    def predict(self, test_features, test_labels):
        self.inherit.predict(test_features, test_labels)
    
    def __str__(self):
        return self.inherit.__str__()