import numpy as np

class GLMRegression():
    def __init__(self,
                 distribution: str,
                 max_iter: int = 100,
                 tol: float = 1e-6) -> None:
        """
        Generalized Linear Model regression supporting multiple distributions via Iteratively Reweighted Least Squares (IRLS)

        Parameters:
            distribution: One of ("gamma", "poisson", "tweedie") indicating which exponential-family distribution to use
            max_iter: Maximum number of IRLS iterations for coefficient convergence
            tol: Convergence tolerance for change in coefficients between iterations
        """
        
        if distribution not in ("gamma", "poisson", "tweedie"):
            raise ValueError("distribution must be 'gamma', 'poisson', or 'tweedie'")
        self.distribution = distribution
        self.max_iter = max_iter
        self.tol = tol
        self._coef = None

    def _inv_link(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute the inverse link function g⁻¹(η) to map linear predictor to mean response

        Parameters:
            eta: Array of linear predictor values 

        Returns:
            np.ndarray: Mean response values μ
        """
        return np.exp(eta)

    def _d_link(self, mu: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the link function g'(μ) with respect to μ

        Parameters:
            mu: Array of mean responses 

        Returns:
            np.ndarray: Derivative values g'(μ)
        """
        return 1.0 / mu

    def _variance(self, mu: np.ndarray) -> np.ndarray:
        """
        Compute the variance function V(μ) for the chosen distribution

        Parameters:
            mu: Array of mean responses 

        Returns:
            np.ndarray: Variance values V(μ)
        """
        if self.distribution == "poisson":
            return mu
        elif self.distribution == "gamma":
            return mu ** 2
        else:  # tweedie with p=1.5 default
            p = 1.5
            return mu ** p

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fit the GLM using Iteratively Reweighted Least Squares (IRLS)

        Parameters:
            features: Training feature matrix of shape 
            targets: True response values of shape
        """
        X = features.squeeze()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        # Add intercept term
        X_design = np.hstack([X, np.ones((n_samples, 1))])

        beta = np.zeros(n_features + 1)

        for iteration in range(self.max_iter):
            eta = X_design @ beta
            # Clip eta to avoid overflow in exp()
            eta = np.clip(eta, -20, 20)  
            mu = self._inv_link(eta)

            # Avoid zeros in mu for division
            mu = np.clip(mu, 1e-8, None)

            g_prime = self._d_link(mu)
            V = self._variance(mu)
            # Avoid zeros in V
            V = np.clip(V, 1e-8, None)

            W = (g_prime ** 2) / V
            z = eta + (targets - mu) * g_prime

            WX = X_design * W[:, np.newaxis]
            beta_new = np.linalg.solve(WX.T @ X_design, WX.T @ z)

            if np.linalg.norm(beta_new - beta) < self.tol:
                beta = beta_new
                break
            beta = beta_new

        self._coef = beta

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict the mean response for new data using the fitted GLM

        Parameters:
            test_features: Feature matrix for prediction of shape 
            test_targets: True response values for evaluation

        Returns:
            np.ndarray: Predicted mean responses of shape
        """
        X = test_features.squeeze()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_design = np.hstack([X, np.ones((X.shape[0], 1))])
        eta = X_design @ self._coef
        eta = np.clip(eta, -20, 20)
        mu = self._inv_link(eta)

        return mu

    def __str__(self) -> str:
        return f"GLM: {self.distribution.capitalize()} Distribution Regression"
