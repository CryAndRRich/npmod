import numpy as np

class tSNE():
    def __init__(self,
                 n_components: int = 2,
                 perplexity: float = 30.0,
                 learning_rate=None,          
                 n_iter: int = 1000,
                 random_state: int = 42,
                 early_exaggeration: float = 12.0,
                 exaggeration_iters: int = 250) -> None:
        """
        Initialize t-SNE model hyperparameters

        Parameters:
            n_components: Target dimension for embedding
            perplexity: Balance between local and global aspects of data
            learning_rate: Step size for gradient updates
            n_iter: Number of optimization iterations
            random_state: Seed for reproducibility
            early_exaggeration: Factor that temporarily increases attractive forces between 
                neighbors during the first 'exaggeration_iters' iterations
            exaggeration_iters: Number of iterations using early exaggeration before switching 
                back to normal probabilities
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.early_exaggeration = early_exaggeration
        self.exaggeration_iters = exaggeration_iters
        self.embedding = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between all pairs of samples"""
        sum_X = np.sum(np.square(X), axis=1)
        D = -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]
        np.fill_diagonal(D, 0.0)
        return D
    
    def _binary_search_perplexity(self, 
                                  distances: np.ndarray, 
                                  tol: float = 1e-5) -> np.ndarray:
        """
        Compute joint probability matrix P using binary search to match a target perplexity

        Parameters:
            distances: Pairwise distance matrix
            tol: Tolerance for entropy difference during binary search

        Returns:
            np.ndarray: Symmetric joint probability matrix
        """
        n = distances.shape[0]
        target = np.log(self.perplexity)
        P = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            beta_min, beta_max = -np.inf, np.inf
            beta = 1.0
            idx = np.concatenate((np.r_[0:i], np.r_[i+1:n]))
            D = distances[i, idx]

            # binary search
            for _ in range(50):
                x = -D * beta
                x -= np.max(x)                
                P_i = np.exp(x)
                sum_P = P_i.sum()
                if sum_P <= 0.0:
                    sum_P = 1e-12
                # Shannon entropy in nats
                H = np.log(sum_P) + beta * np.sum(D * P_i) / sum_P
                H_diff = H - target

                if abs(H_diff) < tol:
                    break
                if H_diff > 0:
                    beta_min = beta
                    beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
                else:
                    beta_max = beta
                    beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

            P[i, idx] = P_i / sum_P

        # Symmetrize & normalize
        P = (P + P.T) / (2.0 * n)
        P = np.maximum(P, 1e-12)
        P /= P.sum()
        return P


    def fit_transform(self, 
                      features: np.ndarray, 
                      pca=None) -> np.ndarray:
        """
        Compute t-SNE embedding of the input features

        Parameters:
            features: Data matrix
            pca: PCA for initialization

        Returns:
            embedding: Low-dimensional representation
        """
        X = features.astype(np.float64, copy=False)
        n, _ = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Heuristic learning rate like sklearn when not specified
        if self.learning_rate is None:
            self.learning_rate_ = max(n / self.early_exaggeration, 50.0)
        else:
            self.learning_rate_ = float(self.learning_rate)

        # Compute high-D probabilities
        D = self._pairwise_distances(X)
        P_true = self._binary_search_perplexity(D)            
        P_eff = P_true * self.early_exaggeration           

        if pca is not None:
            pca.fit(X)
            Y = pca.transform(X)[:, :self.n_components].astype(np.float64)
            # scale down slightly to avoid huge initial distances
            Y = (Y - Y.mean(axis=0)) * 1e-4
        else:
            Y = np.random.randn(n, self.n_components) * 1e-4

        iY = np.zeros_like(Y) # velocity (for momentum)
        momentum, final_momentum = 0.5, 0.8
        momentum_switch_iter = 250

        # optimization loop
        for it in range(1, self.n_iter + 1):
            sum_Y = np.sum(np.square(Y), axis=1)
            # squared distances in low-D:
            dist_Y = -2 * np.dot(Y, Y.T) + sum_Y[:, None] + sum_Y[None, :]
            np.fill_diagonal(dist_Y, 0.0)

            num = 1.0 / (1.0 + dist_Y) # heavy-tailed kernel
            np.fill_diagonal(num, 0.0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # choose which P to use in this phase
            P_used = P_eff if it <= self.exaggeration_iters else P_true

            S = (P_used - Q) * num                           
            srow = np.sum(S, axis=1)                       
            dY = 4.0 * (srow[:, None] * Y - S @ Y)          

            # --- update with momentum ---
            iY = momentum * iY - self.learning_rate_ * dY
            Y += iY

            # recenter to avoid drift
            Y -= Y.mean(axis=0, keepdims=True)

            # switch momentum & stop exaggeration
            if it == momentum_switch_iter:
                momentum = final_momentum
            if it == self.exaggeration_iters:
                P_eff = P_true  # stop early exaggeration

        self.embedding = Y
        return Y

    def __str__(self) -> str:
        return "t-SNE (t-distributed Stochastic Neighbor Embedding)"
