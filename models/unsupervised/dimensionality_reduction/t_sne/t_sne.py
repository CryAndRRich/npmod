import numpy as np

class tSNE():
    def __init__(self, 
                 n_components: int = 2, 
                 perplexity: float = 30.0, 
                 learning_rate: float = 200.0, 
                 n_iter: int = 1000, 
                 random_state: int = 42) -> None:
        """
        Initialize t-SNE model hyperparameters

        Parameters:
            n_components: Target dimension for embedding
            perplexity: Balance between local and global aspects of data
            learning_rate: Step size for gradient updates
            n_iter: Number of optimization iterations
            random_state: Seed for reproducibility
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.embedding = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances between all pairs of samples

        Parameters:
            X: Data matrix

        Returns:
            np.ndarray: Matrix distances between all pairs of samples
        """
        sum_X = np.sum(np.square(X), axis=1)
        return -2 * np.dot(X, X.T) + sum_X[:, np.newaxis] + sum_X[np.newaxis, :]

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
        P = np.zeros((n, n))
        for i in range(n):
            beta_min, beta_max = -np.inf, np.inf
            beta = 1.0
            D = distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            for _ in range(50):
                P_i = np.exp(-D * beta)
                sum_P = np.sum(P_i)
                H = np.log(sum_P) + beta * np.sum(D * P_i) / sum_P
                H_diff = H - target
                if np.abs(H_diff) < tol:
                    break
                if H_diff > 0:
                    beta_min = beta
                    beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
                else:
                    beta_max = beta
                    beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = P_i / sum_P
        return (P + P.T) / (2 * n)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Compute t-SNE embedding of the input features

        Parameters:
            features: Data matrix

        Returns:
            embedding: Low-dimensional representation
        """
        X = features.astype(np.float64)
        n, _ = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)

        dist = self._pairwise_distances(X)
        P = self._binary_search_perplexity(dist)
        P *= 4.0  # Early exaggeration
        P = np.maximum(P, 1e-12)

        Y = np.random.randn(n, self.n_components)
        dY = np.zeros_like(Y)
        momentum = 0.5
        final_momentum = 0.8
        momentum_switch_iter = 250

        for iter in range(self.n_iter):
            sum_Y = np.sum(np.square(Y), axis=1)
            num = 1 / (1 + -2 * np.dot(Y, Y.T) + sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :])
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ = P - Q
            for i in range(n):
                grad = 4 * np.sum((PQ[:, i] * num[:, i])[:, np.newaxis] * (Y[i] - Y), axis=0)
                dY[i] = momentum * dY[i] - self.learning_rate * grad
                Y[i] += dY[i]

            if iter == momentum_switch_iter:
                momentum = final_momentum
            if iter == 100:
                P /= 4.0

        self.embedding = Y
        return Y

    def __str__(self) -> str:
        return "t-distributed Stochastic Neighbor Embedding (t-SNE)"
