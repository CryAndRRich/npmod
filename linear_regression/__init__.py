from .linear_regression_numpy import LinearRegressionNumpy
from .linear_regression_pytorch import LinearRegressionPytorch

class LinearRegression():
    def __init__(self, learn_rate, number_of_epochs, type="numpy"):
        if type == "numpy":
            self.inherit = LinearRegressionNumpy(learn_rate, number_of_epochs)
        elif type == "pytorch":
            self.inherit = LinearRegressionPytorch(learn_rate, number_of_epochs)
        else: 
            raise ValueError(f"Type must be 'numpy' or 'pytorch'")
    
    def fit(self, features, labels):
        self.inherit.fit(features, labels)

    def __str__(self):
        return self.inherit.__str__()