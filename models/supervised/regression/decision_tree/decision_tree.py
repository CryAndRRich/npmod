import numpy as np

class TreeNode():
    def __init__(self,
                 feature: int = None,
                 threshold: float = None,
                 left: "TreeNode" = None,
                 right: "TreeNode" = None,
                 value: float = None) -> None:
        """
        Node in the decision tree for regression

        Attributes:
            feature: Index of the feature used for splitting 
            threshold: Threshold value to split on 
            left: Left child subtree (samples <= threshold)
            right: Right child subtree (samples > threshold)
            value: Predicted value at the leaf node 
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """Return True if this node is a leaf"""
        return self.value is not None


class DecisionTreeRegressor():
    def __init__(self,
                 n_feats: int = None,
                 max_depth: int = 100,
                 min_samples_split: int = 2) -> None:
        """
        Decision Tree Regressor built from scratch

        Parameters:
            n_feats: Number of features to consider when searching for the best split
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split an internal node
        """
        self.root = None
        self.n_feats = n_feats
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self,
            features: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray = None) -> None:
        """
        Build the regression tree using the training data

        Parameters:
            features: Training feature matrix
            targets: Training target values (continuous) array of length n_samples
            sample_weights: Optional 1D array of length n_samples giving each sample's weight
        """
        n_samples, n_features = features.shape
        # Initialize or validate sample weights
        if sample_weights is None:
            sample_weights = np.full(shape=n_samples, fill_value=1.0 / n_samples, dtype=float)
        else:
            sample_weights = sample_weights.astype(float)
            sample_weights /= np.sum(sample_weights)

        # Determine number of features to consider at each split
        self.n_feats = n_features if self.n_feats is None else min(self.n_feats, n_features)

        # Build the tree
        self.root = self.build_tree(features, targets, sample_weights, depth=0)

    def build_tree(self,
                   features: np.ndarray,
                   targets: np.ndarray,
                   sample_weights: np.ndarray,
                   depth: int) -> TreeNode:
        """
        Recursively construct the regression tree

        Parameters:
            features: Feature matrix for current node
            targets: Target vector for current node
            sample_weights: Weights for current node samples
            depth: Current depth in the tree

        Returns:
            TreeNode: Root node of the constructed subtree
        """
        n_samples, n_features = features.shape
        total_weight = np.sum(sample_weights)

        # Compute current node prediction and weighted MSE
        current_pred = float(np.dot(sample_weights, targets) / total_weight)
        current_mse = float(np.dot(sample_weights, (targets - current_pred) ** 2) / total_weight)

        # Stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split or current_mse == 0.0:
            return TreeNode(value=current_pred)

        best_feat_idx = None
        best_threshold = None
        best_mse = current_mse

        # Search for the best split
        for feat_idx in range(n_features):
            if feat_idx >= self.n_feats:
                break
            thresholds = np.unique(features[:, feat_idx])
            for thr in thresholds:
                left_mask = features[:, feat_idx] <= thr
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue

                w_left = sample_weights[left_mask]
                w_right = sample_weights[right_mask]
                y_left = targets[left_mask]
                y_right = targets[right_mask]

                total_left = np.sum(w_left)
                total_right = np.sum(w_right)

                pred_left = float(np.dot(w_left, y_left) / total_left)
                pred_right = float(np.dot(w_right, y_right) / total_right)

                mse_left = float(np.dot(w_left, (y_left - pred_left) ** 2) / total_left)
                mse_right = float(np.dot(w_right, (y_right - pred_right) ** 2) / total_right)

                weighted_mse = (total_left * mse_left + total_right * mse_right) / total_weight

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feat_idx = feat_idx
                    best_threshold = thr

        # If no valid split found, make a leaf
        if best_feat_idx is None:
            return TreeNode(value=current_pred)

        # Partition for recursive calls
        left_mask = features[:, best_feat_idx] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self.build_tree(features[left_mask],
                                       targets[left_mask],
                                       sample_weights[left_mask],
                                       depth + 1)
        
        right_subtree = self.build_tree(features[right_mask],
                                        targets[right_mask],
                                        sample_weights[right_mask],
                                        depth + 1)

        return TreeNode(feature=best_feat_idx,
                        threshold=float(best_threshold),
                        left=left_subtree,
                        right=right_subtree)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for given samples.

        Parameters:
            test_features: Test feature matrix 

        Returns:
            np.ndarray: Predicted target values
        """
        predictions = np.array([
            self.traverse_tree(sample, self.root)
            for sample in test_features
        ])

        return predictions

    def traverse_tree(self,
                      sample: np.ndarray,
                      node: TreeNode) -> float:
        """
        Recursively traverse the tree to predict for one sample

        Parameters:
            sample: Single sample feature vector
            node: Current node in the tree

        Returns:
            float: Predicted value from leaf node
        """
        if node.is_leaf_node():
            return node.value

        if sample[node.feature] <= node.threshold:
            return self.traverse_tree(sample, node.left)
        else:
            return self.traverse_tree(sample, node.right)

    def __str__(self) -> str:
        return "Decision Tree Regressor"
