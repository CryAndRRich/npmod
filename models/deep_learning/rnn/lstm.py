from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class LSTMNet(nn.Module):
    """
    LSTM network module for text classification
    """
    def __init__(self, 
                 embedding: nn.Embedding, 
                 lstm: nn.LSTM, 
                 fc: nn.Linear) -> None:
        super().__init__()
        self.embedding = embedding
        self.lstm = lstm
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        out, (_, _) = self.lstm(x)
        out = out[:, -1, :]  # Take output from last time step
        out = self.fc(out)
        return out

class LSTM(RecurNet):
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 vocab_size: int,
                 embed_dim: int = 100,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 num_classes: int = 10,
                 batch_size: Optional[int] = None) -> None:
        super().__init__(
            learn_rate=learn_rate,
            number_of_epochs=number_of_epochs,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_classes=num_classes,
            batch_size=batch_size
        )
        self.num_layers = num_layers

    def init_network(self) -> None:
        """
        Initialize LSTM network components and training utilities
        """
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        lstm = nn.LSTM(input_size=self.embed_dim,
                       hidden_size=self.hidden_size,
                       num_layers=self.num_layers,
                       batch_first=True)
        fc = nn.Linear(self.hidden_size, self.num_classes)

        self.network = LSTMNet(embedding, lstm, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Long Short-Term Memory (LSTM)"