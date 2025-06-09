import numpy as np

class GaussianMixtureModel():
    def __init__(self,
                 number_of_components: int = 1,
                 max_number_of_epochs: int = 100,
                 tol: float = 1e-6,
                 random_state: int = 42) -> None:
        """
        Gaussian Mixture Model using the EM algorithm

        Parameters:
            number_of_components: Number of Gaussian components (K)
            max_number_of_epochs: Maximum number of EM iterations
            tol: Convergence threshold for log-likelihood change
            random_state: Seed for random initialization
        """
        self.k = number_of_components
        self.max_epochs = max_number_of_epochs
        self.tol = tol
        self.random_state = random_state
        self.means = None      
        self.covariances = None
        self.weights = None   

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

    def _expectation_step(self, features: np.ndarray) -> np.ndarray:
        """
        Expectation step: compute responsibilities

        Parameters:
            features: Data matrix

        Returns:
            gamma: Responsibility matrix
        """
        n_samples, n_features = features.shape
        gamma = np.zeros((n_samples, self.k))
        for i in range(self.k):
            diff = features - self.means[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            denom = np.sqrt(((2 * np.pi) ** n_features) * np.linalg.det(self.covariances[i]))
            exponent = np.einsum('ij,ij->i', diff @ inv_cov, diff)
            pdf = np.exp(-0.5 * exponent) / denom
            gamma[:, i] = self.weights[i] * pdf
        # Normalize responsibilities
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma /= gamma_sum
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
        n_samples, _ = features.shape
        # Effective number of samples per component
        N_k = gamma.sum(axis=0)
        # Update means
        self.means = (gamma.T @ features) / N_k[:, np.newaxis]
        # Update covariances
        for i in range(self.k):
            diff = features - self.means[i]
            weighted = diff.T * gamma[:, i]
            self.covariances[i] = (weighted @ diff) / N_k[i]
        # Update weights
        self.weights = N_k / n_samples

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fits the GMM to the input data

        Parameters:
            features: Data matrix

        Returns:
            labels: Hard cluster assignments based on max responsibility
        """
        self._init_params(features)
        prev_log_likelihood = None
        for epoch in range(1, self.max_epochs + 1):
            gamma = self._expectation_step(features)
            self._maximization_step(features, gamma)
            # Compute log-likelihood
            likelihood = np.zeros(features.shape[0])
            for i in range(self.k):
                diff = features - self.means[i]
                inv_cov = np.linalg.inv(self.covariances[i])
                denom = np.sqrt(((2 * np.pi) ** features.shape[1]) * np.linalg.det(self.covariances[i]))
                exponent = np.einsum('ij,ij->i', diff @ inv_cov, diff)
                pdf = np.exp(-0.5 * exponent) / denom
                likelihood += self.weights[i] * pdf
            log_likelihood = np.sum(np.log(likelihood))
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged at epoch {epoch}/{self.max_epochs}")
                break
            prev_log_likelihood = log_likelihood
        else:
            print(f"Reached max epochs {self.max_epochs}/{self.max_epochs}")
        # Hard assignments
        labels = np.argmax(gamma, axis=1)
        return labels

    def __str__(self) -> str:
        return "Gaussian Mixture Model"
