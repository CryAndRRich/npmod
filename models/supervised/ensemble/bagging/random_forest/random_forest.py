import numpy as np
import random
from ....classification import DecisionTreeClassifier

class RandomForest():
    def __init__(self,
                 n_estimators: int = 100,
                 max_features: str | int | float = "sqrt",
                 max_depth: int | None = None,
                 random_state: int | None = None) -> None:
        """
        Random Forest Classifier
        
        Parameters:
            n_estimators: Number of trees in the forest
            max_features: Number of features to consider at each split
                          - "sqrt": sqrt(n_features)
                          - "log2": log2(n_features)
                          - int: fixed number
                          - float: fraction of features
            max_depth: Maximum depth of each tree (None = unlimited)
            random_state: Seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _get_n_features(self, n_features: int) -> int:
        """Determine number of features to use at each split"""
        if isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            return n_features

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fit the Random Forest model on training data
        
        Parameters:
            features: Feature matrix
            targets: Target labels 
        """
        n_samples, n_features = features.shape
        self.trees = []
        n_feats_per_tree = self._get_n_features(n_features)

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = features[indices], targets[indices]
            
            # Random subset of features
            feat_indices = np.random.choice(n_features, size=n_feats_per_tree, replace=False)
            
            # Build tree
            tree = DecisionTreeClassifier()
            tree.fit(X_sample[:, feat_indices], y_sample)
            self.trees.append((tree, feat_indices))

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in features.
        
        Parameters:
            test_features: Feature matrix (n_samples, n_features)
        
        Returns:
            preds: Predicted labels (n_samples,)
        """
        all_preds = []
        for tree, feat_indices in self.trees:
            preds = tree.predict(test_features[:, feat_indices])
            all_preds.append(preds)
        
        all_preds = np.array(all_preds).T 
        
        # Majority voting
        final_preds = []
        for row in all_preds:
            values, counts = np.unique(row, return_counts=True)
            final_preds.append(values[np.argmax(counts)])
        
        return np.array(final_preds)
    
    def __str__(self) -> str:
        return "Random Forest"
