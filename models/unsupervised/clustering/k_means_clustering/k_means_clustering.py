from typing import List, Tuple
import torch
from ..base import Model

torch.manual_seed(42)

def expectation_step(features: torch.Tensor, 
                     centroids: torch.Tensor, 
                     dists: torch.Tensor, 
                     number_of_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the distance from each feature to each centroid and 
    assigns each feature to the closest centroid

    --------------------------------------------------
    Parameters:
        features: The input data features
        centroids: The current centroids of the clusters
        dists: Preallocated tensor to store distances from each sample to each centroid
        number_of_clusters: The number of clusters

    --------------------------------------------------
    Returns:
        dists_min: The minimum distance of each sample to the centroids
        labels: The label of the closest centroid for each sample
    """
    for i in range(number_of_clusters):  
        ctr = centroids[:, i].unsqueeze(1)
        dists[:, i] = (features - ctr.T).pow(2).sum(dim=1).sqrt()

    dists_min, labels = dists.min(dim=1)
    return dists_min, labels

def maximization_step(features: torch.Tensor, 
                      centroids: torch.Tensor, 
                      labels: torch.Tensor, 
                      number_of_clusters: int) -> torch.Tensor:
    """
    Updates the centroids by computing the mean of all samples assigned to each cluster

    --------------------------------------------------
    Parameters:
        features: The input data features
        centroids: The current centroids of the clusters
        labels: The labels indicating the cluster assignment of each sample
        number_of_clusters: The number of clusters

    --------------------------------------------------
    Returns:
        centroids: The updated centroids
    """
    for i in range(number_of_clusters):  
        ind = torch.where(labels == i)[0]
        if len(ind) == 0:
            continue

        centroids[:, i] = features[ind].mean(dim=0)
    
    return centroids

def arrange(centroids: torch.Tensor, 
            predictions: torch.Tensor) -> Tuple[List, torch.Tensor]:
    """
    Arranges centroids and reassigns cluster labels for consistency

    --------------------------------------------------
    Parameters:
        centroids: The centroids of the clusters
        predictions: The cluster labels for each sample

    --------------------------------------------------
    Returns:
        labeled_centroids: List of centroids with their labels sorted
        predictions: The reassigned cluster labels
    """
    size = centroids.shape[1]

    labeled_centroids = []
    for i in range(size):
        labeled_centroids.append([centroids[0, i].item(), centroids[1, i].item(), i])
    
    labeled_centroids.sort()
    change = {}
    for i, centroid in enumerate(labeled_centroids):
        _, _, ind = centroid
        change[ind] = i + 1
    
    for i in range(predictions.shape[0]):
        predictions[i] = change[predictions[i].item()]
    
    return labeled_centroids, predictions

class KMeansClustering(Model):
    def __init__(self, 
                 number_of_clusters: int = 1, 
                 max_number_of_epochs: int = 20):
        """
        K-Means Clustering model for unsupervised learning

        --------------------------------------------------
        Parameters:
            number_of_clusters: The number of clusters to form
            max_number_of_epochs: The maximum number of iterations to run the algorithm
        """
        self.k = number_of_clusters
        self.max_number_of_epochs = max_number_of_epochs
    
    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Fits the K-Means model to the input data

        --------------------------------------------------
        Parameters:
            features: Feature matrix of the training data
            labels: Array of labels corresponding to the training data
        """
        xmax = features.max(dim=0)[0].unsqueeze(1)
        xmin = features.min(dim=0)[0].unsqueeze(1)
        
        dists = torch.zeros((features.shape[0], self.k))
        centroids = (xmin - xmax) * torch.rand((features.shape[1], self.k)) + xmax
        old_loss = -1

        for epochs in range(1, self.max_number_of_epochs):
            dists_min, predictions = expectation_step(features, centroids, dists, self.k)
            
            centroids = maximization_step(features, centroids, predictions, self.k)
                
            new_loss = dists_min.sum()  
            if old_loss == new_loss:
                print("Stop at epoch: {}/{}".format(epochs, self.max_number_of_epochs))
                break
            old_loss = new_loss
        else:
            print("Done at epoch: {}/{}".format(self.max_number_of_epochs, self.max_number_of_epochs))
        
        self.centroids, predictions = arrange(centroids, predictions)
    
        print("Centroids found:")
        for i, centroid in enumerate(self.centroids):
            x, y, _ = centroid
            print('c{} = ({:.3f}, {:.3f})'.format(i + 1, x, y))

        predictions = predictions.detach().numpy()
        labels = labels.detach().numpy()

        accuracy, f1 = self.evaluate(predictions, labels)
        print("Number of clusters: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.k, accuracy, f1))
    
    def __str__(self):
        return "K Means Clustering"
