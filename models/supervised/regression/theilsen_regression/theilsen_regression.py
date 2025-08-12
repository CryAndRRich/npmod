import random
import numpy as np

class TheilSenRegression():
    def __init__(self) -> None:
        """
        Initializes the Theil-Sen Regression model.
        This model is robust to outliers and estimates slope using the median of pairwise slopes
        """
        self.slope = None
        self.intercept = None

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray,
            max_pairs: int = 100000) -> None:
        """
        Fits the Theil-Sen regression model using the training data

        Parameters:
            features: The input features for training 
            targets: The target targets corresponding to the input features 
        """
        n_samples, _ = features.shape
        betas = []

        max_possible_pairs = n_samples * (n_samples - 1) // 2
        num_pairs = min(max_pairs, max_possible_pairs)

        pairs = set()
        while len(pairs) < num_pairs:
            i, j = random.sample(range(n_samples), 2)
            if i > j:
                i, j = j, i
            pairs.add((i, j))

        for i, j in pairs:
            diff_x = features[j] - features[i]
            if np.all(diff_x == 0):
                continue
            diff_y = targets[j] - targets[i]
            beta_ij = diff_y / np.linalg.norm(diff_x) ** 2 * diff_x
            betas.append(beta_ij)

        betas = np.array(betas)
        if betas.size == 0:
            raise ValueError("Cannot compute slopes: all features differences are zero")

        self.slope = np.median(betas, axis=0)
        self.intercept = np.median(targets - features @ self.slope)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the trained Theil-Sen regression model

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        predictions = test_features @ self.slope + self.intercept

        return predictions

    def __str__(self) -> str:
        return "Theil-Sen Regression"
