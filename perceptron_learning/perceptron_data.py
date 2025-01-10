import numpy as np
import torch
from sklearn.model_selection import train_test_split

class PerceptronData():
    def __init__(self, data_path):
        self.NUMPY_DATA = np.genfromtxt(data_path, delimiter = ',', skip_header=1)

        self.X_NUMPY, self.Y_NUMPY = self.NUMPY_DATA[:, :-1], self.NUMPY_DATA[:, -1]
        self.X_NUMPY_TRAIN, self.X_NUMPY_TEST, self.Y_NUMPY_TRAIN, self.Y_NUMPY_TEST = train_test_split(self.X_NUMPY, self.Y_NUMPY, test_size=0.2, random_state=42)

        self.X_TENSOR_TRAIN = torch.FloatTensor(self.X_NUMPY_TRAIN)
        self.X_TENSOR_TEST = torch.FloatTensor(self.X_NUMPY_TEST)
        self.Y_TENSOR_TRAIN = torch.FloatTensor(self.Y_NUMPY_TRAIN).unsqueeze(1)
        self.Y_TENSOR_TEST = torch.FloatTensor(self.Y_NUMPY_TEST).unsqueeze(1)

        self.LEARN_RATE = 0.0001
        self.NUMBER_OF_EPOCHS = 12
    
    def get_numpy_data(self):
        return self.X_NUMPY_TRAIN, self.X_NUMPY_TEST, self.Y_NUMPY_TRAIN, self.Y_NUMPY_TEST
    
    def get_tensor_data(self):
        return self.X_TENSOR_TRAIN, self.X_TENSOR_TEST, self.Y_TENSOR_TRAIN, self.Y_TENSOR_TEST
    
    def get_parameters(self):
        return self.LEARN_RATE, self.NUMBER_OF_EPOCHS
