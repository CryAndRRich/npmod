import torch
from base_model import ModelML

torch.manual_seed(42)

def expectation_step(features, centroids, dists, number_of_clusters):
    for i in range(number_of_clusters):  
        ctr = centroids[:, i].unsqueeze(1)
        dists[:, i] = (features - ctr.T).pow(2).sum(dim=1).sqrt()

    dists_min, labels = dists.min(dim=1)
    return dists_min, labels

def maximization_step(features, centroids, labels, number_of_clusters):
    for i in range(number_of_clusters):  
        ind = torch.where(labels == i)[0]
        if len(ind) == 0:
            continue

        centroids[:, i] = features[ind].mean(dim=0)
    
    return centroids

def arrange(centroids, predictions):
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


class KMeansClustering(ModelML):
    def __init__(self, number_of_clusters=1, max_number_of_epochs=20):
        self.k = number_of_clusters
        self.max_number_of_epochs = max_number_of_epochs
    
    def fit(self, features, labels):
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
        print("k: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.k, accuracy, f1))
    
    def __str__(self):
        return "K Means Clustering"