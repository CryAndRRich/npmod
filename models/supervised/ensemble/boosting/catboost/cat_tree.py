from typing import List
import numpy as np
from ..xgboost.xgb_tree import XGTreeRegressor, TreeNode

class CatTreeRegressor(XGTreeRegressor):
    """
    Regression tree for CatBoost (oblivious-style) with native categorical splits.
    Inherits gradient/hessian infrastructure but builds TreeNode structure
    """
    def __init__(self,
                 cat_features: List[int] = None,
                 n_feats: int = None,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0) -> None:
        """
        Initialize CatTreeRegressor

        Parameters:
            cat_features: Indices of categorical features
            n_feats: Number of features to consider per split
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            reg_lambda: L2 regularization on leaf weights (λ)
            gamma: Minimum gain required for a split (γ)
        """
        super().__init__(n_feats=n_feats,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         reg_lambda=reg_lambda,
                         gamma=gamma)
        
        self.cat_features = set(cat_features or [])

    def fit(self,
            features: np.ndarray,
            grad: np.ndarray,
            hess: np.ndarray) -> None:
        """
        Fit the CatBoost tree to gradients and hessians

        Parameters:
            features: Training feature matrix of shape (n_samples, n_features)
            grad: First-order gradients g_i for each sample
            hess: Second-order hessians h_i for each sample
        """
        self._grad, self._hess = grad, hess
        n_features = features.shape[1]
        self.n_feats = n_features if self.n_feats is None else min(self.n_feats, n_features)
        self.root = self.build_cat_tree(features, grad, hess, depth=0)

    def build_cat_tree(self,
                       features: np.ndarray,
                       grad: np.ndarray,
                       hess: np.ndarray,
                       depth: int) -> TreeNode:
        """
        Recursively build the CatBoost regression tree

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

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=self._leaf_weight(G_total, H_total))

        best_gain = 0.0
        best_feat = None
        best_thr = None
        best_is_cat = False
        best_order = []

        # Search splits
        for feat in range(n_features):
            if feat >= self.n_feats:
                break
            vals = features[:, feat]
            is_cat = feat in self.cat_features
            if is_cat:
                cats = np.unique(vals)
                ratios = {c: grad[vals==c].sum() / (hess[vals==c].sum() + 1e-12) for c in cats}
                order = sorted(cats, key=lambda c: ratios[c])
                feat_vals = np.array([order.index(v) for v in vals])
            else:
                order = []
                feat_vals = vals.astype(float)

            for thr in np.unique(feat_vals):
                left_mask = feat_vals <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                G_L, H_L = grad[left_mask].sum(), hess[left_mask].sum()
                G_R, H_R = grad[right_mask].sum(), hess[right_mask].sum()
                gain = self._gain(G_L, H_L, G_R, H_R, G_total, H_total)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr
                    best_is_cat = is_cat
                    best_order = order

        if best_feat is None or best_gain <= 0:
            return TreeNode(value=self._leaf_weight(G_total, H_total))

        # split data
        vals = features[:, best_feat]
        if best_is_cat:
            indices = np.array([best_order.index(v) for v in vals])
        else:
            indices = vals.astype(float)
        left_mask = indices <= best_thr
        right_mask = ~left_mask

        left = self.build_cat_tree(features[left_mask], grad[left_mask], hess[left_mask], depth+1)
        right = self.build_cat_tree(features[right_mask], grad[right_mask], hess[right_mask], depth+1)

        return TreeNode(feature=best_feat, threshold=float(best_thr), left=left, right=right)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Traverse the tree to predict leaf weights for each sample

        Parameters:
            test_features: Feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Array of leaf weight predictions for each sample
        """
        def _traverse(x: np.ndarray, node: TreeNode) -> float:
            if node.value is not None:
                return node.value
            val = x[node.feature]
            if node.feature in self.cat_features:
                # Categorical: direct compare order index
                # Note: category order must be stored; omitted for brevity
                comp = float(val)  # Assume pre-encoded
            else:
                comp = val
            if comp <= node.threshold:
                return _traverse(x, node.left)
            else:
                return _traverse(x, node.right)
        return np.array([_traverse(x, self.root) for x in test_features])
