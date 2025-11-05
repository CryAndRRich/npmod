import torch
import torch.nn as nn
import torch.optim as optim

class RecurNet():
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 input_size: int,
                 hidden_size: int = 128,
                 output_size: int = 1,
                 forecast_horizon: int = 1,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 dropout: float = 0.1) -> None:
        """
        Base Recurrent Neural Network class for sequence classification

        Parameters:
            learn_rate: Learning rate for the optimizer
            number_of_epochs: Number of training epochs
            input_size: Size of the input features
            hidden_size: Number of hidden units in the RNN layer
            output_size: Number of output classes
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of RNN layers
            bidirectional: If True, use a bidirectional RNN
            dropout: Dropout rate between RNN layers
        """

        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

    def init_network(self) -> None:
        """Initialize network, optimizer, criterion"""
        pass

    def init_weights(self, m) -> None:
        """Initialize the model parameters using the Xavier initializer"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        """
        Trains the network on the training set
        
        Parameters:
            train_loader: The DataLoader for training data
            verbose: If True, prints training progress
        """
        self.init_network()
        self.network.apply(self.init_weights)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.MSELoss()

        self.network.train()
        for epoch in range(self.number_of_epochs):
            total_loss = 0.0
            for features, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.network(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * features.size(0)

            avg_loss = total_loss / len(train_loader.dataset)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}], Loss: {avg_loss:.4f}")

    def predict(self, test_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Makes predictions on the test set and evaluates the network

        Parameters:
            test_loader: The DataLoader for testing

        Returns:
            predictions: The prediction targets
        """
        self.network.eval()
        all_preds = []

        with torch.no_grad():
            for features in test_loader:
                if isinstance(features, (list, tuple)):
                    features = features[0]

                preds = self.network(features)
                all_preds.append(preds)
        
        predictions = torch.cat(all_preds, dim=0)

        return predictions

from .LSTM import LSTM
from .GRU import GRU
from .MGU import MGU
from .UGRNN import UGRNN
from .RHN import RHN
from .SRU import SRU
from .JANET import JANET
from .IndRNN import IndRNN
from .RAN import RAN
from .SCRN import SCRN
from .YamRNN import YamRNN