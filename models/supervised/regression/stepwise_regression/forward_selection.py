import numpy as np
from ..linear_regression import LinearRegression
from .utils import *

class StepwiseForward():
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 criterion: str = "mse",
                 threshold_in: float = 1e-4,
                 verbose: bool = False) -> None:
        """
        Forward stepwise selection using specified criterion

        Parameters:
            learn_rate: Learning rate for gradient descent
            number_of_epochs: Epochs per trial fit
            criterion: One of "mse", "r2", "aic"
            threshold_in: Minimum improvement in criterion to add
            verbose: Print progress
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.criterion = criterion
        self.threshold_in = threshold_in
        self.verbose = verbose
        self.selected_features = []

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
            return -r2  # Invert to minimize
        if self.criterion == "aic":
            k = X.shape[1] + 1
            return compute_aic(len(y), mse, k)
        raise ValueError("Criterion must be mse, r2, or aic")

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Perform forward stepwise selection and train the final model on selected features

        Parameters:
            features: Input feature matrix
            targets: True targets
        """

        _, n_features = features.shape
        remaining = list(range(n_features))
        # Initialize best score
        if self.criterion == "r2":
            best_score = float("-inf")
        else:
            best_score = float("inf")

        self.selected_features = []

        while remaining:
            best_candidate = None
            candidate_score = best_score

            for feat in remaining:
                cols = self.selected_features + [feat]
                X_sub = features[:, cols]
                model = LinearRegression(self.learn_rate, self.number_of_epochs)
                model.fit(X_sub, targets)
                score = self._evaluate(model, X_sub, targets)

                if self.verbose:
                    print(f"Trying feature {feat}: {self.criterion} = {score:.5f}")

                improved = (score + self.threshold_in < candidate_score)
                if improved:
                    candidate_score = score
                    best_candidate = feat

            if best_candidate is None:
                break

            self.selected_features.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = candidate_score
            if self.verbose:
                print(f"Added feature {best_candidate}, new {self.criterion} = {best_score:.5f}")

        self.model = LinearRegression(self.learn_rate, self.number_of_epochs)
        X_final = features[:, self.selected_features]
        self.model.fit(X_final, targets)

    def predict(self, test_features: np.ndarray):
        """
        Predict and optionally evaluate on test data

        Parameters:
            test_features: Test feature matrix 

        Returns:
            np.ndarray: Predicted target values
        """
        X_sub = test_features[:, self.selected_features]
        return self.model.predict(X_sub)

    def __str__(self) -> str:
        return "Stepwise Regression: Forward Selection"