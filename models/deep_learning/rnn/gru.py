from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class GRUNet(nn.Module):
    """
    GRU network module for sequence classification
    """
    def __init__(self,
                 embedding: nn.Embedding,
                 gru: nn.GRU,
                 fc: nn.Linear) -> None:
        super().__init__()
        self.embedding = embedding
        self.gru = gru
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class GRU(RecurNet):
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 vocab_size: int,
                 embed_dim: int = 100,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 bidirectional: bool = False,
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
        self.bidirectional = bidirectional

    def init_network(self) -> None:
        embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        gru = nn.GRU(input_size=self.embed_dim,
                     hidden_size=self.hidden_size,
                     num_layers=self.num_layers,
                     batch_first=True,
                     bidirectional=self.bidirectional)
        fc_in = self.hidden_size * (2 if self.bidirectional else 1)
        fc = nn.Linear(fc_in, self.num_classes)

        self.network = GRUNet(embedding, gru, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Gated Recurrent Units (GRU)"
