import numpy as np
import torch

class KMeansData():
    def __init__(self, data_path):
        self.NUMPY_DATA = np.genfromtxt(data_path, delimiter = ',', skip_header=1)

        self.NUMPY_TRAIN, self.NUMPY_LABEL = self.NUMPY_DATA[:, :-1], self.NUMPY_DATA[:, -1]

        self.TENSOR_TRAIN = torch.FloatTensor(self.NUMPY_TRAIN)
        self.TENSOR_LABEL = torch.FloatTensor(self.NUMPY_LABEL).unsqueeze(1)

        self.MAX_NUMBER_OF_EPOCHS = 100
    
    def get_numpy_data(self):
        return self.NUMPY_TRAIN, self.NUMPY_LABEL
    
    def get_tensor_data(self):
        return self.TENSOR_TRAIN, self.TENSOR_LABEL
    
    def get_parameters(self):
        return self.MAX_NUMBER_OF_EPOCHS
    

