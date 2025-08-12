import numpy as np

class SupportVectorMachine:
    def __init__(self, 
                 kernel: str = "rbf", 
                 C: float = 1.0,
                 gamma: float = 0.1, 
                 degree: int = 3, 
                 coef0: float = 1.0,
                 tol: float = 1e-3,
                 max_iter: int = 1000):
        """
        Support Vector Machine (binary classification)

        Parameters:
            kernel: "linear", "rbf", "poly", or "sigmoid"
            C: Regularization parameter
            gamma: Kernel coefficient (used in RBF, poly, sigmoid)
            degree: Degree for polynomial kernel
            coef0: Constant term for poly and sigmoid
            tol: Tolerance for stopping
            max_iter: Maximum training iterations
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = None
        self.bias = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.features = None

        # Assign kernel function
        if kernel == "linear":
            self.kernel = self._linear
        elif kernel == "rbf":
            self.kernel = self._rbf
        elif kernel == "poly":
            self.kernel = self._poly
        elif kernel == "sigmoid":
            self.kernel = self._sigmoid
        else:
            raise ValueError("Unsupported")

    def _linear(self, x, y):
        return np.dot(x, y.T)

    def _rbf(self, x, y):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if y.ndim == 1:
            y = y[np.newaxis, :]
        sq_dists = np.sum((x[:, np.newaxis] - y[np.newaxis, :]) ** 2, axis=2)
        return np.exp(-self.gamma * sq_dists)

    def _poly(self, x, y):
        return (self.gamma * np.dot(x, y.T) + self.coef0) ** self.degree

    def _sigmoid(self, x, y):
        return np.tanh(self.gamma * np.dot(x, y.T) + self.coef0)

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Train SVM

        Parameters:
            features: Training data (n_samples, n_features)
            y: Labels in {-1, 1}
        """
        n_samples, _ = features.shape
        targets = targets.astype(float)
        targets[targets == 0] = -1  # Convert 0 to -1

        self.features = features
        self.alphas = np.zeros(n_samples)
        self.bias = 0

        K = self.kernel(features, features)  # Kernel matrix

        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alphas)

            for i in range(n_samples):
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)

                yi, yj = targets[i], targets[j]

                kii = K[i, i]
                kjj = K[j, j]
                kij = K[i, j]

                eta = kii + kjj - 2 * kij
                if eta <= 0:
                    continue

                Ei = np.sum(self.alphas * targets * K[:, i]) + self.bias - yi
                Ej = np.sum(self.alphas * targets * K[:, j]) + self.bias - yj

                alpha_j_new = self.alphas[j] + yj * (Ei - Ej) / eta
                alpha_j_new = np.clip(alpha_j_new, 0, self.C)

                alpha_i_new = self.alphas[i] + yi * yj * (self.alphas[j] - alpha_j_new)

                self.alphas[i] = alpha_i_new
                self.alphas[j] = alpha_j_new

            diff = np.linalg.norm(self.alphas - alpha_prev)
            if diff < self.tol:
                break

        # Extract support vectors
        idx = self.alphas > 1e-5
        self.support_vectors = features[idx]
        self.support_vector_labels = targets[idx]
        self.support_vector_alphas = self.alphas[idx]

        # Compute bias 
        self.bias = np.mean(
            self.support_vector_labels - 
            np.sum(self.support_vector_alphas * self.support_vector_labels *
                   self.kernel(self.support_vectors, self.support_vectors), axis=0)
        )

    def _decision_function(self, x):
        K = self.kernel(self.support_vectors, x)
        result = np.sum(self.support_vector_alphas * self.support_vector_labels * K) + self.bias
        return result

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data

        Parameters:
            test_features: Input features

        Returns:
            Array of 0/1 predictions
        """
        decision = np.array([self._decision_function(x) for x in test_features])
        return np.where(decision >= 0, 1, 0)
