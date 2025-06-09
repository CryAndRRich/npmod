import numpy as np

class BayesianLinearRegression():
    def __init__(self,
                 sigma2: float = 1.0,
                 tau2: float = 1.0) -> None:
        """
        Bayesian Linear Regression with a Gaussian prior and Gaussian likelihood

        Parameters:
            sigma2: Noise variance in the Gaussian likelihood
            tau2: Prior variance for weight parameters
        """
        self.sigma2 = sigma2
        self.tau2 = tau2
        self.w_mean = None
        self.w_cov = None

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Fit the Bayesian linear regression model by computing the posterior over weights

        Parameters:
            features: Training feature matrix 
            targets: Target vector 
        """
        # Add bias term by appending a column of ones
        X = features.squeeze()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        X_design = np.hstack([X, np.ones((n_samples, 1))])

        # Prior covariance and precision
        prior_cov = self.tau2 * np.eye(n_features + 1)
        prior_pre = np.linalg.inv(prior_cov)

        # Likelihood precision
        lik_pre = (1.0 / self.sigma2) * (X_design.T @ X_design)

        # Posterior precision and covariance
        post_pre = prior_pre + lik_pre
        self.w_cov = np.linalg.inv(post_pre)

        # Posterior mean
        self.w_mean = (1.0 / self.sigma2) * self.w_cov @ X_design.T @ targets

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict target values for test data, optionally returning predictive standard deviation

        Parameters:
            test_features: Test feature matrix 

        Returns:
            np.ndarray: Predicted means
        """
        X_test = test_features.squeeze()
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        n_samples, _ = X_test.shape
        X_design = np.hstack([X_test, np.ones((n_samples, 1))])

        # Predictive mean
        predictions = X_design @ self.w_mean  

        return predictions

    def __str__(self) -> str:
        return "Bayesian Linear Regression"
