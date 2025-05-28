import torch
import torch.nn as nn
from typing import Optional

class RecurNet():
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 vocab_size: int,
                 embed_dim: int = 100,
                 hidden_size: int = 128,
                 num_classes: int = 10,
                 batch_size: Optional[int] = None):
        """
        Base Recurrent Neural Network class for sequence classification

        Parameters:
            learn_rate: Learning rate for optimizer
            number_of_epochs: Number of training epochs
            vocab_size: Vocabulary size of the dataset
            embed_dim: Dimension of word embeddings
            hidden_size: Number of hidden units in GRU layer
            num_classes: Number of output classes for classification
            batch_size: Batch size used during training
        """

        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size

    def init_network(self) -> None:
        """Initialize network, optimizer, criterion"""

    def init_weights(self, m) -> None:
        """Initialize the model parameters using the Xavier initializer"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def fit(self, 
            features: torch.Tensor, 
            targets: torch.Tensor) -> None:
        """
        Train the network using the provided dataset

        Parameters:
            features: Input tensor
            targets: Target class indices
        """
        self.init_network()
        self.network.train()

        for _ in range(self.number_of_epochs):
            self.optimizer.zero_grad()
            outputs = self.network(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def predict(self, test_features: torch.Tensor) -> torch.Tensor:
        """
        Predict classes for test features

        Parameters:
            test_features: Input tensor 

        Returns:
            predictions: Tensor of predicted class indices 
        """
        self.network.eval()
        with torch.no_grad():
            logits = self.network(test_features)
            predictions = torch.argmax(logits, dim=1)

        return predictions
    
from .lstm import LSTM
from .gru import GRU
from .indrnn import IndRNN
from .janet import JANET
from .mgu import MGU
from .ran import RAN
from .rhn import RHN
from .scrn import SCRN
from .sru import SRU
from .ugrnn import UGRNN
from .yamrnn import YamRNN