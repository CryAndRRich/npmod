from typing import Optional
import numpy as np
from .gaussian_nb import GaussianNaiveBayes
from .multinomial_nb import MultinomialNaiveBayes
from .bernoulli_nb import BernoulliNaiveBayes
from .categorical_nb import CategoricalNaiveBayes

class NaiveBayes():
    def __init__(self, 
                 distribution: str = "gaussian", 
                 alpha: Optional[int] = None) -> None:
        """
        Initializes the Naive Bayes model by selecting the appropriate distribution type

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
            targets: np.ndarray) -> None:
        
        self.inherit.fit(features, targets)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        
        predictions = self.inherit.predict(test_features)
        return predictions

    def __str__(self) -> str:
        return self.inherit.__str__()
