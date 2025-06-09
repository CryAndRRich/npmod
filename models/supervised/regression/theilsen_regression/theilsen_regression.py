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
            targets: np.ndarray) -> None:
        """
        Fits the Theil-Sen regression model using the training data

        Parameters:
            features: The input features for training 
            targets: The target targets corresponding to the input features 
        """
        x = features.flatten()
        y = targets.flatten()
        n = len(x)

        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                if x[j] != x[i]:
                    slopes.append((y[j] - y[i]) / (x[j] - x[i]))

        self.slope = np.median(slopes)
        self.intercept = np.median(y - self.slope * x)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the trained Theil-Sen regression model

        Parameters:
            test_features: Test feature matrix

        Returns:
            np.ndarray: Predicted target values
        """
        predictions = self.slope * test_features + self.intercept

        return predictions

    def __str__(self) -> str:
        return "Theil-Sen Regression"
