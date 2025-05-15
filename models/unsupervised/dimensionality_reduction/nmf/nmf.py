import numpy as np

class NMF():
    def __init__(self, 
                 n_components: int, 
                 max_iter: int = 200, 
                 tol: float = 1e-4, 
                 random_state: int = 42) -> None:
        """
        Initialize Non-negative Matrix Factorization model

        Parameters:
            n_components: Number of latent components r
            max_iter: Maximum number of update iterations
            tol: Tolerance for convergence based on reconstruction error change
            random_state: Seed for reproducibility of W and H initialization
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.components_ = None  # H matrix of shape (n_components, n_features)
        self.transformer_ = None  # W matrix of shape (n_samples, n_components)
        self.reconstruction_err_ = None

    def fit(self, features: np.ndarray) -> None:
        """
        Fit the NMF model to the non-negative input data

        Parameters:
            features: Input data matrix, all values >= 0
        """
        X = features.astype(np.float64)
        if np.any(X < 0):
            raise ValueError("Input matrix must be non-negative")
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Initialize W and H with non-negative random values
        W = rng.rand(n_samples, self.n_components)
        H = rng.rand(self.n_components, n_features)

        prev_err = None
        for iteration in range(self.max_iter):
            # Update H
            numerator = W.T.dot(X)
            denominator = W.T.dot(W).dot(H) + 1e-10
            H *= numerator / denominator

            # Update W
            numerator = X.dot(H.T)
            denominator = W.dot(H.dot(H.T)) + 1e-10
            W *= numerator / denominator

            # Compute reconstruction error
            X_approx = W.dot(H)
            err = np.linalg.norm(X - X_approx, 'fro')
            if prev_err is not None and abs(prev_err - err) < self.tol:
                break
            prev_err = err

        self.transformer_ = W
        self.components_ = H
        self.reconstruction_err_ = err

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform data into the latent component space W

        Parameters:
            features: New data matrix, all values >= 0

        Returns:
            W_new: Encoding matrix
        """
        if self.components_ is None:
            raise ValueError("NMF model is not fitted yet. Call fit() before transform().")
        X = features.astype(np.float64)
        if np.any(X < 0):
            raise ValueError("Input matrix X must be non-negative")

        # Initialize W_new
        m_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        W_new = rng.rand(m_samples, self.n_components)

        # Fix H = components_, update W_new only
        H = self.components_
        for iteration in range(self.max_iter):
            numerator = X.dot(H.T)
            denominator = W_new.dot(H.dot(H.T)) + 1e-10
            W_new *= numerator / denominator
        return W_new

    def __str__(self) -> str:
        return "Non-negative Matrix Factorization (NMF)"
