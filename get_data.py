from typing import Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class getData():
    """
    Data processing class for preparing datasets for machine learning models

    This class supports handling numeric and text data, providing both NumPy and 
    PyTorch tensor formats (not for text data) for training and testing
    """
    def __init__(self, 
                 data_path: str, 
                 data_type: str = "number"):
        """
        Initializes the data processing object by loading the dataset and preparing features and labels

        Parameters:
        data_path: Path to the CSV dataset file
        data_type: The type of data to process ('number' for numerical data, 'text' for text data)
        """
        self.data = pd.read_csv(data_path, lineterminator='\n')
        self.features, self.labels = self.data.iloc[:, :-1], self.data.iloc[:, -1]

        if data_type == "number":
            self.numpy_features = self.features.to_numpy()
            self.numpy_labels = self.labels.to_numpy()
            
            # Split data into training and testing sets for numerical data
            self.numpy_train_features, self.numpy_test_features, \
            self.numpy_train_labels, self.numpy_test_labels = train_test_split(
                self.numpy_features,
                self.numpy_labels,
                test_size=0.2,
                random_state=42
            )

            # Convert the split data into PyTorch tensors
            self.tensor_train_features = torch.FloatTensor(self.numpy_train_features)
            self.tensor_test_features = torch.FloatTensor(self.numpy_test_features)
            self.tensor_train_labels = torch.LongTensor(self.numpy_train_labels)
            self.tensor_test_labels = torch.LongTensor(self.numpy_test_labels)

        elif data_type == "text":
            self.numpy_labels = self.labels.to_numpy()
            
            # Split data into training and testing sets for text data
            self.numpy_train_features, self.numpy_test_features, \
            self.numpy_train_labels, self.numpy_test_labels = train_test_split(
                self.features,
                self.numpy_labels,
                test_size=0.2,
                random_state=42
            )

            # Convert a collection of text documents to a matrix of token counts using CountVectorizer()
            vectorizer = CountVectorizer()
            self.numpy_train_features = vectorizer.fit_transform(self.numpy_train_features['text']).toarray()
            self.numpy_test_features = vectorizer.transform(self.numpy_test_features['text']).toarray()

    def get_processed_data(self, type_of_data: str = "numpy") \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | \
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves processed training and testing datasets in the specified format

        Parameters:
        type_of_data: The format of data to return ('numpy' for NumPy arrays, 'tensor' for PyTorch tensors)

        Returns:
        Tuple: The processed datasets in the specified format:
            - Training features
            - Testing features
            - Training labels
            - Testing labels
        """
        if type_of_data == "numpy":
            return (self.numpy_train_features,
                    self.numpy_test_features,
                    self.numpy_train_labels,
                    self.numpy_test_labels)
        
        elif type_of_data == "tensor":
            return (self.tensor_train_features,
                    self.tensor_test_features,
                    self.tensor_train_labels,
                    self.tensor_test_labels)
    
        else:
            raise ValueError(f"Type of data must be 'numpy' or 'tensor'")
