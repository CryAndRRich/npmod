import numpy as np

class BayesianGaussianMixtureModel():
    def __init__(self,
                 number_of_components: int = 1,
                 max_number_of_epochs: int = 100,
                 tol: float = 1e-6,
                 random_state: int = 42,
                 alpha_prior: float = 1.0) -> None:
        """
        Bayesian Gaussian Mixture Model using Variational Inference

        Parameters:
            number_of_components: Number of Gaussian components (K)
            max_number_of_epochs: Maximum number of VI iterations
            tol: Convergence threshold for ELBO change
            random_state: Seed for reproducibility
            alpha_prior: Dirichlet prior parameter for mixture weights
        """
        self.k = number_of_components
        self.max_epochs = max_number_of_epochs
        self.tol = tol
        self.random_state = random_state
        self.alpha_prior = alpha_prior

        # Variational parameters
        self.means = None
        self.covariances = None
        self.weights = None
        self.alpha = None  # Dirichlet parameters

    def _init_params(self, features: np.ndarray) -> None:
        n_samples, n_features = features.shape
        np.random.seed(self.random_state)

        # Initialize means by selecting random samples
        idxs = np.random.choice(n_samples, self.k, replace=False)
        self.means = features[idxs]
        # Initialize covariances to identity
        self.covariances = np.array([np.eye(n_features) for _ in range(self.k)])
        # Initialize weights uniformly
        self.weights = np.full(self.k, 1 / self.k)
        # Dirichlet parameters for weights
        self.alpha = np.full(self.k, self.alpha_prior + n_samples / self.k)

    def _expectation_step(self, features: np.ndarray) -> np.ndarray:
        """
        Expectation step: compute responsibilities under variational parameters

        Parameters:
            features: Data matrix

        Returns:
            gamma: Responsibility matrix (n_samples x k)
        """
        n_samples, n_features = features.shape
        gamma = np.zeros((n_samples, self.k))
        for i in range(self.k):
            diff = features - self.means[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            denom = np.sqrt(((2 * np.pi) ** n_features) * np.linalg.det(self.covariances[i]))
            exponent = np.einsum('ij,ij->i', diff @ inv_cov, diff)
            pdf = np.exp(-0.5 * exponent) / (denom + 1e-12)
            gamma[:, i] = (self.alpha[i] / np.sum(self.alpha)) * pdf

        # Normalize responsibilities
        gamma_sum = np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)
        gamma /= gamma_sum
        return gamma

    def _maximization_step(self,
                           features: np.ndarray,
                           gamma: np.ndarray) -> None:
        """
        Maximization step: update variational parameters

        Parameters:
            features: Data matrix
            gamma: Responsibility matrix
        """
        _, n_features = features.shape
        N_k = gamma.sum(axis=0)

        # Update means
        self.means = (gamma.T @ features) / np.clip(N_k[:, np.newaxis], 1e-12, None)

        # Update covariances with regularization
        for i in range(self.k):
            diff = features - self.means[i]
            weighted = diff.T * gamma[:, i]
            self.covariances[i] = (weighted @ diff) / np.clip(N_k[i], 1e-12, None)
            self.covariances[i] += 1e-6 * np.eye(n_features)

        # Update Dirichlet parameters
        self.alpha = self.alpha_prior + N_k

        # Update weights as expectation under Dirichlet
        self.weights = self.alpha / np.sum(self.alpha)

    def fit(self, 
            features: np.ndarray, 
            return_gamma: bool = False) -> np.ndarray:
        """
        Fits the Bayesian GMM to the input data using Variational Inference

        Parameters:
            features: Data matrix
            return_gamma: If True, also return responsibilities

        Returns:
            labels: Hard cluster assignments
            gamma (optional): Responsibility matrix
        """
        self._init_params(features)
        prev_elbo = None

        for epoch in range(1, self.max_epochs + 1):
            gamma = self._expectation_step(features)
            self._maximization_step(features, gamma)

            # Compute Evidence Lower Bound (ELBO) approx = log-likelihood under q
            likelihood = np.zeros(features.shape[0])
            for i in range(self.k):
                diff = features - self.means[i]
                inv_cov = np.linalg.inv(self.covariances[i])
                denom = np.sqrt(((2 * np.pi) ** features.shape[1]) *
                                np.linalg.det(self.covariances[i]))
                exponent = np.einsum('ij,ij->i', diff @ inv_cov, diff)
                pdf = np.exp(-0.5 * exponent) / (denom + 1e-12)
                likelihood += self.weights[i] * pdf
            elbo = np.sum(np.log(likelihood + 1e-12))

            if prev_elbo is not None and abs(elbo - prev_elbo) < self.tol:
                print(f"Converged at epoch {epoch}/{self.max_epochs}")
                break
            prev_elbo = elbo
        else:
            print(f"Reached max epochs {self.max_epochs}/{self.max_epochs}")

        labels = np.argmax(gamma, axis=1)
        if return_gamma:
            return labels, gamma
        return labels

    def __str__(self) -> str:
        return "Bayesian Gaussian Mixture Model"
