import numpy as np
from ....regression.decision_tree.decision_tree import *

class XGTreeRegressor(DecisionTreeRegressor):
    """
    Regression tree specialized for XGBoost (uses gradient and hessian)
    """
    def __init__(self,
                 n_feats: int = None,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0) -> None:
        """
        Initialize the XGBoost regression tree

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
        Fit the XGBoost tree to gradients and hessians

        Parameters:
            features: Training feature matrix of shape (n_samples, n_features)
            grad: First-order gradients g_i for each sample
            hess: Second-order hessians h_i for each sample
        """
        # store gradients and hessians
        self._grad = grad
        self._hess = hess
        n_features = features.shape[1]
        self.n_feats = n_features if self.n_feats is None else min(self.n_feats, n_features)
        self.root = self.build_xgb_tree(features, grad, hess, depth=0)

    def _gain(self, 
              G_left: float, 
              H_left: float, 
              G_right: float, 
              H_right: float, 
              G_total: float, 
              H_total: float) -> float:
        """
        Compute the split gain based on aggregated gradients and hessians

        Parameters:
            G_left: Sum of gradients for left split
            H_left: Sum of hessians for left split
            G_right: Sum of gradients for right split
            H_right: Sum of hessians for right split
            G_total: Sum of gradients at parent node (G_left + G_right)
            H_total: Sum of hessians at parent node (H_left + H_right)

        Returns:
            float: Gain value = 0.5 * [G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G_T^2/(H_T+λ)] - γ
        """
        term_left = (G_left ** 2) / (H_left + self.reg_lambda)
        term_right = (G_right ** 2) / (H_right + self.reg_lambda)
        term_total = (G_total ** 2) / (H_total + self.reg_lambda)
        return 0.5 * (term_left + term_right - term_total) - self.gamma

    def _leaf_weight(self, 
                     G: float, 
                     H: float) -> float:
        """
        Calculate the optimal weight for a leaf node

        Parameters:
            G: Sum of gradients for samples at this leaf
            H: Sum of hessians for samples at this leaf

        Returns:
            float: Leaf weight = -G / (H + λ).
        """
        return - G / (H + self.reg_lambda)

    def build_xgb_tree(self,
                       features: np.ndarray,
                       grad: np.ndarray,
                       hess: np.ndarray,
                       depth: int) -> TreeNode:
        """
        Recursively build the XGBoost regression tree

        Parameters:
            features: Feature matrix at current node
            grad: Gradients for current node samples
            hess: Hessians for current node samples
            depth: Current depth in the tree

        Returns:
            TreeNode: Root of the constructed subtree
        """

        n_samples, n_features = features.shape
        G_total, H_total = grad.sum(), hess.sum()

        # if stopping, return leaf
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            val = self._leaf_weight(G_total, H_total)
            return TreeNode(value=val)

        best_gain = 0.0
        best_feat = None
        best_thr = None
        best_masks = None

        # iterate features
        for feat in range(n_features):
            if feat >= self.n_feats:
                break
            thresholds = np.unique(features[:, feat])
            for thr in thresholds:
                left_mask = features[:, feat] <= thr
                right_mask = ~left_mask
                if not (left_mask.any() and right_mask.any()):
                    continue
                G_L, H_L = grad[left_mask].sum(), hess[left_mask].sum()
                G_R, H_R = grad[right_mask].sum(), hess[right_mask].sum()
                gain = self._gain(G_L, H_L, G_R, H_R, G_total, H_total)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr
                    best_masks = (left_mask, right_mask)

        # no split or no gain
        if best_feat is None:
            val = self._leaf_weight(G_total, H_total)
            return TreeNode(value=val)

        left_mask, right_mask = best_masks
        left = self.build_xgb_tree(features[left_mask], grad[left_mask], hess[left_mask], depth+1)
        right = self.build_xgb_tree(features[right_mask], grad[right_mask], hess[right_mask], depth+1)
        return TreeNode(feature=best_feat, threshold=float(best_thr), left=left, right=right)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Traverse the tree to predict leaf weights for each sample

        Parameters:
            test_features: Feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Array of leaf weight predictions for each sample
        """
        # traverse tree, returning leaf weight
        def _pred(x: np.ndarray, 
                  node: TreeNode) -> float:
            if node.value is not None:
                return node.value
            if x[node.feature] <= node.threshold:
                return _pred(x, node.left)
            else:
                return _pred(x, node.right)
            
        return np.array([_pred(x, self.root) for x in test_features])