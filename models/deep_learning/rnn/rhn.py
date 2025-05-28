from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class RHNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 depth: int = 2) -> None:
        """
        Initialize the RHN cell

        Parameters:
            input_size: Size of the input vector at each time step
            hidden_size: Number of hidden units
            depth: Number of highway layers in the cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth

        self.input_transform = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(depth)
        ])
        self.transform_gates = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(depth)
        ])

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one time step of the RHN cell

        Parameters:
            x_t: Input tensor at current time step
            h_prev: Previous hidden state

        Returns:
            h: Current hidden state
        """
        h = self.input_transform(x_t)

        for i in range(self.depth):
            t_gate = torch.sigmoid(self.transform_gates[i](h))
            h_layer = torch.tanh(self.layers[i](h))
            h = t_gate * h_layer + (1 - t_gate) * h

        return h + h_prev  # Residual connection

class RHNNet(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding, 
                 cell: RHNCell, 
                 fc: nn.Linear) -> None:
        """
        Initialize the RHN classification network

        Parameters:
            embedding: Embedding layer for input tokens
            cell: RHN recurrent cell
            fc: Fully connected layer for final classification
        """
        super().__init__()
        self.embedding = embedding
        self.cell = cell
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RHN network

        Parameters:
            x: Input tensor
            
        Returns:
            out: Logits tensor
        """
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.shape
        h = torch.zeros(batch_size, self.cell.hidden_size, device=embedded.device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h = self.cell(x_t, h)

        out = self.fc(h)
        return out

class RHN(RecurNet):
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 vocab_size: int,
                 embed_dim: int = 100,
                 hidden_size: int = 128,
                 num_classes: int = 10,
                 depth: int = 2,
                 batch_size: Optional[int] = None) -> None:
        """
        Initialize the RHN training model

        Parameters:
            learn_rate: Learning rate for optimizer
            number_of_epochs: Number of training epochs
            vocab_size: Size of the input vocabulary
            embed_dim: Size of the word embedding vectors
            hidden_size: Number of hidden units in RHN cell
            num_classes: Number of output classes
            depth: Number of highway layers in each RHN cell
            batch_size: Batch size used during training
        """
        super().__init__(
            learn_rate=learn_rate,
            number_of_epochs=number_of_epochs,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_classes=num_classes,
            batch_size=batch_size
        )
        self.depth = depth

    def init_network(self) -> None:
        """
        Initialize RHNNet model, weights, loss function, and optimizer
        """
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        cell = RHNCell(input_size=self.embed_dim, hidden_size=self.hidden_size, depth=self.depth)
        fc = nn.Linear(self.hidden_size, self.num_classes)

        self.network = RHNNet(embedding, cell, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Recurrent Highway Network (RHN)"
