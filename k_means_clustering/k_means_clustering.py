import torch

class KMeansClustering():
    def __init__(self, number_of_clusters=1):
        self.k = number_of_clusters
        self.centroids = None
    
    def expectation_step(self, x_train, centroids, dists, number_of_clusters):
        for i in range(number_of_clusters):  
            ctr = centroids[:, i].unsqueeze(1)
            dists[:, i] = (x_train - ctr.T).pow(2).sum(dim=1).sqrt()

        dists_min, labels = dists.min(dim=1)
        return dists_min, labels
    
    def maximization_step(self, x_train, centroids, labels, number_of_clusters):
        for i in range(number_of_clusters):  
            ind = torch.where(labels == i)[0]
            if len(ind) == 0:
                continue

            centroids[:, i] = x_train[ind].mean(dim=0)
        
        return centroids

def train_model(x_train, labels_train, number_of_clusters, max_number_of_epochs=20):
    kms = KMeansClustering(number_of_clusters)

    xmax = x_train.max(dim=0)[0].unsqueeze(1)
    xmin = x_train.min(dim=0)[0].unsqueeze(1)
    
    dists = torch.zeros((x_train.shape[0], kms.k))
    centroids = (xmin - xmax) * torch.rand((x_train.shape[1], kms.k)) + xmax
    old_loss = -1

    for epochs in range(max_number_of_epochs + 1):
        dists_min, labels_predict = kms.expectation_step(x_train, centroids, dists, number_of_clusters)
        
        centroids = kms.maximization_step(x_train, centroids, labels_predict, number_of_clusters)
            
        new_loss = dists_min.sum()  
        if old_loss == new_loss:
            print('Stop at epoch: {}/{}'.format(epochs + 1, max_number_of_epochs))
            break
        old_loss = new_loss
    else:
        print('Done at epoch: {}/{}'.format(max_number_of_epochs, max_number_of_epochs))
    
    labeled_centroids, correct_labels = test_model(centroids, labels_predict, labels_train)
    print('Centroids found:')
    for i, centroid in enumerate(labeled_centroids):
        x, y, _ = centroid
        print('c{} = ({:.3f}, {:.3f})'.format(i + 1, x, y))
    
    print('Correctly Labeled: {}/{} Accuracy: {}%'.format(correct_labels, labels_predict.shape[0], correct_labels / labels_predict.shape[0] * 100))

def test_model(centroids, labels_predict, label_train):
    size = centroids.shape[1]

    labeled_centroids = []
    for i in range(size):
        labeled_centroids.append([centroids[0, i].item(), centroids[1, i].item(), i])
    
    labeled_centroids.sort()
    change = {}
    for i, centroid in enumerate(labeled_centroids):
        _, _, ind = centroid
        change[ind] = i + 1
    
    correct_labels = 0
    for i in range(labels_predict.shape[0]):
        correct_labels += int(label_train[i] == change[labels_predict[i].item()])
    
    return labeled_centroids, correct_labels
