import numpy as np
import torch

class PolynomialRegressionData():
    def __init__(self, data_path):
        self.NUMPY_DATA = np.genfromtxt(data_path, delimiter = ',', skip_header=1)

        self.X_NUMPY_TRAIN, self.Y_NUMPY_TRAIN = self.NUMPY_DATA[:, 0], self.NUMPY_DATA[:, 1]

        self.X_TENSOR_TRAIN = torch.FloatTensor(self.X_NUMPY_TRAIN).unsqueeze(1)
        self.Y_TENSOR_TRAIN = torch.FloatTensor(self.Y_NUMPY_TRAIN).unsqueeze(1)

        self.LEARN_RATE = 0.00001
        self.BATCH_SIZE = self.X_NUMPY_TRAIN.shape[0]
        self.NUMBER_OF_EPOCHS = 100000
    
    def get_numpy_data(self):
        return self.X_NUMPY_TRAIN, self.Y_NUMPY_TRAIN
    
    def get_tensor_data(self):
        return self.X_TENSOR_TRAIN, self.Y_TENSOR_TRAIN
    
    def get_parameters(self):
        return self.LEARN_RATE, self.BATCH_SIZE, self.NUMBER_OF_EPOCHS