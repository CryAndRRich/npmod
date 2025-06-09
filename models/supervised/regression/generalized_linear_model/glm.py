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

    def _link(self, mu: np.ndarray) -> np.ndarray:
        """
        Compute the link function g(μ) for the chosen distribution

        Parameters:
            mu: Array of mean responses 

        Returns:
            np.ndarray: Transformed linear predictor values η = g(μ).
        """
        if self.distribution in ("poisson", "tweedie"):
            return np.log(mu)
        # Gamma uses inverse link: η = 1/μ
        return 1.0 / mu

    def _inv_link(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute the inverse link function g⁻¹(η) to map linear predictor to mean response

        Parameters:
            eta: Array of linear predictor values 

        Returns:
            np.ndarray: Mean response values μ
        """
        if self.distribution in ("poisson", "tweedie"):
            return np.exp(eta)
        # Gamma: μ = 1/η
        return 1.0 / eta

    def _d_link(self, mu: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the link function g'(μ) with respect to μ

        Parameters:
            mu: Array of mean responses 

        Returns:
            np.ndarray: Derivative values g'(μ)
        """
        if self.distribution in ("poisson", "tweedie"):
            return 1.0 / mu
        # Gamma: derivative of g(μ) = 1/μ is -1/μ²
        return -1.0 / (mu ** 2)

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
        if self.distribution == "gamma":
            return mu ** 2
        # Tweedie: variance function μ^p with p default to 1.5
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
        # Initialize coefficients
        beta = np.zeros(n_features + 1)

        for iteration in range(self.max_iter):
            # Compute linear predictor and mean response
            eta = self._link(np.clip(X_design @ beta, 1e-8, None))
            mu = self._inv_link(eta)

            # Compute derivative of link and variance
            g_prime = self._d_link(mu)
            V = self._variance(mu)

            # Compute weights W and working response z for IRLS
            W = (g_prime ** 2) / V
            z = eta + (targets - mu) * g_prime

            # IRLS update: beta_new = (X^T W X)^(-1) X^T W z
            WX = X_design * W[:, np.newaxis]
            beta_new = np.linalg.solve(WX.T @ X_design, WX.T @ z)

            # Check convergence
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
        mu = self._inv_link(eta)

        return mu

    def __str__(self) -> str:
        return f"GLM: {self.distribution.capitalize()} Distribution Regression"
