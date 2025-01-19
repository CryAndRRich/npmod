import numpy as np

def entropy(labels, weights=None):
    if weights is None:
        weights = np.ones(len(labels))
    weighted_counts = np.bincount(labels, weights=weights)
    total_weight = np.sum(weights)
    probs = weighted_counts / total_weight
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
    return entropy

def split_data(features, labels, feature, value, weights=None):
    if isinstance(value, (int, float)):
        true_indices = np.where(features[:, feature] <= value)[0]
        false_indices = np.where(features[:, feature] > value)[0]
    else:  
        true_indices = np.where(features[:, feature] == value)[0]
        false_indices = np.where(features[:, feature] != value)[0]

    true_features, true_labels = features[true_indices], labels[true_indices]
    false_features, false_labels = features[false_indices], labels[false_indices]

    if weights is None:
        return true_features, true_labels, false_features, false_labels
    
    true_weights, false_weights = weights[true_indices], weights[false_indices]
    return true_features, true_labels, true_weights, false_features, false_labels, false_weights

def gain_ratio(true_labels, false_labels, current_entropy, true_weights=None, false_weights=None):
    true = len(true_labels) if true_weights is None else np.sum(true_weights)
    false = len(false_labels) if false_weights is None else np.sum(false_weights)

    total_size = true + false
    true_ratio = true / total_size
    false_ratio = false / total_size
    
    true_entropy = entropy(true_labels, true_weights)
    false_entropy = entropy(false_labels, false_weights)
    
    information_gain = current_entropy - (true_ratio * true_entropy + false_ratio * false_entropy)
    split_info = - (true_ratio * np.log2(true_ratio) + false_ratio * np.log2(false_ratio)) if true_ratio > 0 and false_ratio > 0 else 0
    
    return information_gain / split_info if split_info != 0 else 0