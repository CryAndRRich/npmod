import numpy as np
from numpy.linalg import svd, norm
from typing import Optional, Tuple

class NMF():
    def __init__(self,
                 n_components: int,
                 max_iter: int = 200,
                 tol: float = 1e-4,
                 init: str = "nndsvda",
                 beta_loss: str = "frobenius",
                 l1_reg_W: float = 0.0,
                 l1_reg_H: float = 0.0,
                 random_state: int = 42,
                 normalize: bool = True,
                 clip_eps: float = 1e-10) -> None:
        """
        Initialize Non-negative Matrix Factorization model

        Parameters:
            n_components: Rank of the factorization
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            init: Initialization method for W and H, {"random", "nndsvd", "nndsvda"}
            beta_loss: Only Frobenius is implemented 
            l1_reg_W, l1_reg_H: Small L1 regularization to mitigate scale explosion
            random_state: Seed for reproducibility
            normalize: If True, normalize columns of W to unit L2 after each full MU step
            clip_eps: Small epsilon to avoid division by zero and negative drift due to FP error
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.beta_loss = beta_loss
        self.l1_reg_W = l1_reg_W
        self.l1_reg_H = l1_reg_H
        self.random_state = random_state
        self.normalize = normalize
        self.clip_eps = clip_eps

        self.components_ = None
        self.transformer_ = None

    def fit(self, features: np.ndarray) -> None:
        """
        Fit the NMF model to the non-negative input data

        Parameters:
            features: Input data matrix, all values >= 0
        """
        X = self._validate_X(features)
        W, H = self._init_WH(X)

        prev_loss = np.inf
        for _ in range(1, self.max_iter + 1):
            # Update H
            WT = W.T
            numerator = WT @ X
            denominator = (WT @ W) @ H + self.l1_reg_H + self.clip_eps
            H *= numerator / denominator
            H = np.maximum(H, 0.0)  # numerical safety

            # Update W
            HT = H.T
            numerator = X @ HT
            denominator = W @ (H @ HT) + self.l1_reg_W + self.clip_eps
            W *= numerator / denominator
            W = np.maximum(W, 0.0)

            loss = norm(X - W @ H, 'fro')
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.transformer_ = W
        self.components_ = H

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform data into the latent component space W

        Parameters:
            features: New data matrix, all values >= 0

        Returns:
            W: Encoding matrix
        """
        if self.components_ is None:
            raise ValueError("NMF model is not fitted yet. Call fit() first.")
        
        X = self._validate_X(features)
        rng = np.random.RandomState(self.random_state)
        W = rng.rand(X.shape[0], self.n_components)
        W = np.maximum(W, self.clip_eps)

        H = self.components_

        for _ in range(self.max_iter):
            HT = H.T
            numerator = X @ HT
            denominator = W @ (H @ HT) + self.l1_reg_W + self.clip_eps
            W *= numerator / denominator
            W = np.maximum(W, 0.0)

        return W

    def inverse_transform(self, W: Optional[np.ndarray] = None) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("NMF model is not fitted yet.")
        if W is None:
            if self.transformer_ is None:
                raise ValueError("No stored W. Pass W explicitly or call fit() first.")
            W = self.transformer_
        return W @ self.components_
    
    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D array.")
        if np.any(X < 0):
            raise ValueError("Input matrix must be non-negative.")
        return X

    def _init_WH(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_features = X.shape
        r = self.n_components
        rng = np.random.RandomState(self.random_state)

        if self.init == "random":
            W = rng.rand(n_samples, r)
            H = rng.rand(r, n_features)
            W = np.maximum(W, self.clip_eps)
            H = np.maximum(H, self.clip_eps)
            if self.normalize:
                W, H = self._normalize_columns(W, H)
            return W, H
        elif self.init in ("nndsvd", "nndsvda"):
            W, H = self._nndsvd(X, r, variant=self.init)
            if self.normalize:
                W, H = self._normalize_columns(W, H)
            return W, H
        else:
            raise ValueError("init must be 'random', 'nndsvd', or 'nndsvda'.")

    def _normalize_columns(self, W: np.ndarray, H: np.ndarray, rescale_H: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize columns of W to unit L2 norm; rescale rows of H accordingly
        col_norms = np.maximum(norm(W, axis=0), self.clip_eps)
        Wn = W / col_norms
        if rescale_H:
            Hn = H * col_norms[:, None]
            return Wn, Hn
        else:
            return Wn, H

    def _nndsvd(self, X: np.ndarray, r: int, variant: str = "nndsvd") -> Tuple[np.ndarray, np.ndarray]:
        """
        NNDSVD initialization following Boutsidis et al. (2008).
        variant: 'nndsvd' for exact zeros; 'nndsvda' fills zeros with small positive values.
        """
        U, S, Vt = svd(X, full_matrices=False)
        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]

        W = np.zeros((X.shape[0], r))
        H = np.zeros((r, X.shape[1]))

        # First component: nonnegative parts of first singular triplet
        W[:, 0] = np.maximum(U[:, 0], 0)
        H[0, :] = np.maximum(Vt[0, :], 0) * S[0]

        for j in range(1, r):
            u = U[:, j]
            v = Vt[j, :]
            u_p = np.maximum(u, 0)
            u_n = np.maximum(-u, 0)
            v_p = np.maximum(v, 0)
            v_n = np.maximum(-v, 0)

            u_p_norm = norm(u_p)
            v_p_norm = norm(v_p)
            u_n_norm = norm(u_n)
            v_n_norm = norm(v_n)

            term_p = u_p_norm * v_p_norm
            term_n = u_n_norm * v_n_norm

            if term_p >= term_n:
                W[:, j] = u_p / max(u_p_norm, self.clip_eps)
                H[j, :] = v_p / max(v_p_norm, self.clip_eps) * S[j]
            else:
                W[:, j] = u_n / max(u_n_norm, self.clip_eps)
                H[j, :] = v_n / max(v_n_norm, self.clip_eps) * S[j]

        if variant == "nndsvda":
            # Fill zeros with small positive values to avoid zero-locking
            avg = X.mean()
            eps = 1e-6 * avg if avg > 0 else 1e-6
            W[W == 0] = eps
            H[H == 0] = eps
        return W, H

    def __str__(self) -> str:
        return "NMF (Non-negative Matrix Factorization)"

