import numpy as np
from ..linear_regression import LinearRegression
from .utils import *

class StepwiseBackward():
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 criterion: str = "mse",
                 threshold_out: float = 1e-4,
                 verbose: bool = False) -> None:
        """
        Initialize a backward elimination instance

        Parameters:
            learn_rate: Learning rate for gradient descent in each submodel
            number_of_epochs: Number of training epochs per submodel
            criterion: One of "mse", "r2", "aic" to guide elimination
            threshold_out: Minimum improvement in the criterion to remove a feature
            verbose: If True, prints trial results for each candidate removal
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.criterion = criterion
        self.threshold_out = threshold_out
        self.verbose = verbose
        self.selected_feature = []

    def _evaluate(self, 
                  model: LinearRegression, 
                  X: np.ndarray, 
                  y: np.ndarray) -> float:
        """
        Evaluate the trained model on given data according to the chosen criterion

        Parameters:
            model: Trained Linear Regression instance
            X: Feature matrix used for evaluation
            y: True targets

        Returns:
            float: Score to minimize
        """
        preds = model.predict(X)
        mse = compute_mse(preds, y)
        if self.criterion == "mse":
            return mse
        r2 = compute_r2(preds, y)
        if self.criterion == "r2":
            return -r2
        k = X.shape[1] + 1
        return compute_aic(len(y), mse, k)

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Perform backward elimination and train the final model on retained features

        Parameters:
            features: Input feature matrix
            targets: True targets
        """
        _, n_features = features.shape
        self.selected_features = list(range(n_features))

        # Compute initial full model score
        model_full = LinearRegression(self.learn_rate, self.number_of_epochs)
        model_full.fit(features, targets)
        best_score = self._evaluate(model_full, features, targets)

        while len(self.selected_features) > 1:
            worst_candidate = None
            candidate_score = best_score

            for feat in self.selected_features:
                cols = [f for f in self.selected_features if f != feat]
                X_sub = features[:, cols]
                model = LinearRegression(self.learn_rate, self.number_of_epochs)
                model.fit(X_sub, targets)
                score = self._evaluate(model, X_sub, targets)
                if self.verbose:
                    print(f"Trying remove feature {feat}: {self.criterion} = {score:.5f}")
                improved = (score + self.threshold_out < candidate_score)
                if improved:
                    candidate_score = score
                    worst_candidate = feat

            if worst_candidate is None:
                break
            self.selected_features.remove(worst_candidate)
            best_score = candidate_score
            if self.verbose:
                print(f"Removed feature {worst_candidate}, new {self.criterion} = {best_score:.5f}")

        # Train final model on retained features
        self.model = LinearRegression(self.learn_rate, self.number_of_epochs)
        X_final = features[:, self.selected_features]
        self.model.fit(X_final, targets)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict and optionally evaluate on test data

        Parameters:
            test_features: Test feature matrix 

        Returns:
            np.ndarray: Predicted target values.
        """
        X_sub = test_features[:, self.selected_features]
        return self.model.predict(X_sub)

    def __str__(self) -> str:
        return "Stepwise Regression: Backward Elimination"