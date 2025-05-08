import numpy as np
from ....regression.decision_tree.decision_tree import *

class XGTreeRegressor(DecisionTreeRegressor):
    def __init__(self,
                 n_feats: int = None,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0) -> None:
        """
        Initialize the regression tree

        Parameters:
            n_feats: Number of features to consider when looking for the best split
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split an internal node
            reg_lambda: L2 regularization term on leaf weights (λ)
            gamma: Minimum loss reduction required to make a split (γ)
        """
        super().__init__(n_feats=n_feats,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split)
        
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        
    def fit(self,
            features: np.ndarray,
            grad: np.ndarray,
            hess: np.ndarray) -> None:
        """
        Fit the tree to gradients and hessians

        Parameters:
            features: Training feature matrix of shape (n_samples, n_features)
            grad: First-order gradients g_i for each sample
            hess: Second-order hessians h_i for each sample
        """
        self._grad = grad
        self._hess = hess
        n_features = features.shape[1]
        self.n_feats = n_features if self.n_feats is None else min(self.n_feats, n_features)
        self.root = self._build_tree(features, grad, hess, depth=0)

    def _build_tree(self,
                    features: np.ndarray,
                    grad: np.ndarray,
                    hess: np.ndarray,
                    depth: int) -> TreeNode:
        
        """
        Recursively build the regression tree

        Parameters:
            features: Feature matrix at current node
            grad: Gradients for current node samples
            hess: Hessians for current node samples
            depth: Current depth in the tree

        Returns:
            TreeNode: Root of the constructed subtree
        """
        n_samples, _ = features.shape
        G_T, H_T = grad.sum(), hess.sum()

        # Stopping
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=self._leaf_weight(G_T, H_T))

        best = {"gain": 0.0}
        # Try every split
        for feat in range(features.shape[1]):
            if feat >= self.n_feats:
                break
            for thr in np.unique(features[:,feat]):
                left = features[:,feat] <= thr
                right = ~left
                if not (left.any() and right.any()):
                    continue
                G_L, H_L = grad[left].sum(), hess[left].sum()
                G_R, H_R = grad[right].sum(), hess[right].sum()
                gain = self._gain(G_L, H_L, G_R, H_R, G_T, H_T)
                if gain > best["gain"]:
                    best = dict(gain=gain, feat=feat, thr=thr,
                                left_mask=left, right_mask=right)

        if "feat" not in best:
            return TreeNode(value=self._leaf_weight(G_T, H_T))

        left_node = self._build_tree(features[best["left_mask"]],
                                     grad[best["left_mask"]],
                                     hess[best["left_mask"]],
                                     depth+1)
        right_node = self._build_tree(features[best["right_mask"]],
                                      grad[best["right_mask"]],
                                      hess[best["right_mask"]],
                                      depth+1)

        return TreeNode(feature=best["feat"],
                        threshold=float(best["thr"]),
                        left=left_node,
                        right=right_node)

    def predict(self, features):
        def _walk(x, node):
            if node.value is not None:
                return node.value
            if x[node.feature] <= node.threshold:
                return _walk(x, node.left)
            return _walk(x, node.right)

        return np.array([_walk(x, self.root) for x in features])

    def _gain(self, 
              G_L: float, 
              H_L: float, 
              G_R: float, 
              H_R: float, 
              G_T: float, 
              H_T: float) -> float:
        """
        Compute the split gain based on aggregated gradients and hessians

        Parameters:
            G_L: Sum of gradients for left split
            H_L: Sum of hessians for left split
            G_R: Sum of gradients for right split
            H_R: Sum of hessians for right split
            G_T: Sum of gradients at parent node (G_L + G_R)
            H_T: Sum of hessians at parent node (H_L + H_R)

        Returns:
            float: Gain value = 0.5 * [G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G_T^2/(H_T+λ)] - γ
        """
        left = G_L ** 2 / (H_L + self.reg_lambda)
        right = G_R ** 2 / (H_R + self.reg_lambda)
        total = G_T ** 2 / (H_T + self.reg_lambda)
        return 0.5 * (left + right - total) - self.gamma

    def _leaf_weight(self, 
                     G: float, 
                     H: float) -> float:
        """
        Calculate the optimal weight for a leaf node

        Parameters:
            G: Sum of gradients for samples at this leaf
            H: Sum of hessians for samples at this leaf

        Returns:
            float: Leaf weight = -G / (H + λ)
        """
        return -G / (H + self.reg_lambda)
