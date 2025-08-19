import numpy as np

class GaussianMixtureModel():
    def __init__(self,
                 number_of_components: int = 1,
                 max_number_of_epochs: int = 100,
                 tol: float = 1e-6,
                 random_state: int = 42,
                 reg_covar: float = 1e-6) -> None:
        """
        Gaussian Mixture Model using the EM algorithm

        Parameters:
            number_of_components: Number of Gaussian components (K)
            max_number_of_epochs: Maximum number of EM iterations
            tol: Convergence threshold for log-likelihood change
            random_state: Seed for random initialization
            reg_covar: Regularization added to diagonal of covariances
        """
        self.k = number_of_components
        self.max_epochs = max_number_of_epochs
        self.tol = tol
        self.random_state = random_state
        self.reg_covar = reg_covar

        self.means = None
        self.covariances = None
        self.weights = None

    def _init_params(self, features: np.ndarray) -> None:
        n_samples, n_features = features.shape
        rng = np.random.default_rng(self.random_state)

        # Initialize means by selecting random samples
        idxs = rng.choice(n_samples, self.k, replace=False)
        self.means = features[idxs]
        # Initialize covariances to identity
        self.covariances = np.array([np.eye(n_features) for _ in range(self.k)])
        # Initialize weights uniformly
        self.weights = np.full(self.k, 1 / self.k)

    def _estimate_pdf(self, features: np.ndarray) -> np.ndarray:
        """Compute Gaussian pdf for each component and each sample."""
        n_samples, n_features = features.shape
        pdfs = np.zeros((n_samples, self.k))
        for i in range(self.k):
            diff = features - self.means[i]
            cov = self.covariances[i]
            inv_cov = np.linalg.inv(cov)
            denom = np.sqrt(((2 * np.pi) ** n_features) * np.linalg.det(cov))
            exponent = np.einsum('ij,ij->i', diff @ inv_cov, diff)
            pdfs[:, i] = np.exp(-0.5 * exponent) / denom
        return pdfs

    def _expectation_step(self, features: np.ndarray) -> np.ndarray:
        """
        Expectation step: compute responsibilities (gamma)

        Parameters:
            features: Data matrix

        Returns:
            gamma: Responsibility matrix
        """
        pdfs = self._estimate_pdf(features)
        weighted_pdfs = pdfs * self.weights
        gamma_sum = np.clip(weighted_pdfs.sum(axis=1, keepdims=True), 1e-12, None)
        gamma = weighted_pdfs / gamma_sum
        return gamma

    def _maximization_step(self, 
                           features: np.ndarray, 
                           gamma: np.ndarray) -> None:
        """
        Maximization step: update parameters based on responsibilities

        Parameters:
            features: Data matrix
            gamma: Responsibility matrix
        """
        n_samples, n_features = features.shape
        # Effective number of samples per component
        N_k = gamma.sum(axis=0)
        # Update means
        self.means = (gamma.T @ features) / N_k[:, np.newaxis]
        # Update covariances
        for i in range(self.k):
            diff = features - self.means[i]
            weighted = diff.T * gamma[:, i]
            cov = (weighted @ diff) / N_k[i]
            # Regularization to avoid singular matrix
            self.covariances[i] = cov + self.reg_covar * np.eye(n_features)
        # Update weights
        self.weights = N_k / n_samples

    def fit(self, features: np.ndarray, return_soft: bool = False):
        """
        Fits the GMM to the input data

        Parameters:
            features: Data matrix
            return_soft: If True, also return responsibility matrix (soft assignment)

        Returns:
            labels: Hard cluster assignments
            gamma (optional): Responsibility matrix if return_soft=True
        """
        self._init_params(features)
        prev_log_likelihood = None

        for epoch in range(1, self.max_epochs + 1):
            gamma = self._expectation_step(features)
            self._maximization_step(features, gamma)

            # Compute log-likelihood
            pdfs = self._estimate_pdf(features)
            likelihood = (pdfs * self.weights).sum(axis=1)
            log_likelihood = np.sum(np.log(np.clip(likelihood, 1e-12, None)))

            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged at epoch {epoch}/{self.max_epochs}")
                break
            prev_log_likelihood = log_likelihood
        else:
            print(f"Reached max epochs {self.max_epochs}/{self.max_epochs}")

        # Hard assignments
        labels = np.argmax(gamma, axis=1)
        return (labels, gamma) if return_soft else labels

    def __str__(self) -> str:
        return "Gaussian Mixture Model"
