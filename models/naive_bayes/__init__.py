import numpy as np
from .gaussian_nb import GaussianNaiveBayes
from .multinomial_nb import MultinomialNaiveBayes
from .bernoulli_nb import BernoulliNaiveBayes
from .categorical_nb import CategoricalNaiveBayes

class NaiveBayes():
    def __init__(self, 
                 distribution: str = "gaussian", 
                 alpha: int = 1) -> None:
        """
        Initializes the Naive Bayes model by selecting the appropriate distribution type

        --------------------------------------------------
        Parameters:
            distribution: Type of Naive Bayes distribution ("gaussian", "multinomial", "bernoulli", "categorical")
            alpha: Smoothing parameter for models that require Laplace smoothing
        """
        # Set the appropriate model based on the distribution type
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

    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray) -> None:
        
        self.inherit.fit(features, labels)

    def predict(self, 
                test_features: np.ndarray, 
                test_labels: np.ndarray,
                get_accuracy: bool = True) -> np.ndarray:
        
        predictions = self.inherit.predict(test_features, test_labels, get_accuracy)
        return predictions

    def __str__(self) -> str:
        return self.inherit.__str__()
