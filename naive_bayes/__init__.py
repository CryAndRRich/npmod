from .gaussian_nb import GaussianNaiveBayes
from .multinomial_nb import MultinomialNaiveBayes
from .bernoulli_nb import BernoulliNaiveBayes
from .categorical_nb import CategoricalNaiveBayes

class NaiveBayes():
    def __init__(self, distribution="gaussian", alpha=1):
        if distribution == "gaussian":
            self.inherit = GaussianNaiveBayes()
        elif distribution == "multinomial":
            self.inherit = MultinomialNaiveBayes(alpha=alpha)
        elif distribution == "bernoulli":
            self.inherit = BernoulliNaiveBayes(alpha=alpha)
        elif distribution == "categorical":
            self.inherit = CategoricalNaiveBayes(alpha=alpha)
        else: 
            raise ValueError(f"Unsupported distribution '{distribution}'. Supported types of distribution are 'gaussian', 'multinomial', 'bernoulli', and 'categorical'")
    
    def fit(self, features, labels):
        self.inherit.fit(features, labels)

    def predict(self, test_features, test_labels):
        self.inherit.predict(test_features, test_labels)
    
    def __str__(self):
        return self.inherit.__str__()