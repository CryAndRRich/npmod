import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class UGRNNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize the UGRNN cell

        Parameters:
            input_size: Size of the input vector at each time step
            hidden_size: Size of the hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single time step

        Parameters:
            x_t: Input tensor at time step t 
            h_prev: Hidden state from the previous time step 

        Returns:
            h_t: Updated hidden state 
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        z_t = torch.sigmoid(self.W_z(combined))
        h_tilde = torch.tanh(self.W_h(combined))
        h_t = z_t * h_prev + (1 - z_t) * h_tilde
        return h_t

class UGRNNNet(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding, 
                 cell: UGRNNCell, 
                 fc: nn.Linear) -> None:
        """
        Initialize the UGRNN network

        Parameters:
            embedding: Embedding layer to convert token indices into vectors
            cell: An instance of UGRNNCell
            fc: Final linear layer for output classification
        """
        super().__init__()
        self.embedding = embedding
        self.cell = cell
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass over an input sequence

        Parameters:
            x: Input tensor 

        Returns:
            out: Logits for each class 
        """
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.shape
        h = torch.zeros(batch_size, self.cell.hidden_size, device=embedded.device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h = self.cell(x_t, h)

        out = self.fc(h)
        return out

class UGRNN(RecurNet):
    def init_network(self) -> None:
        """
        Initialize network layers, loss function, and optimizer
        """
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        cell = UGRNNCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
        fc = nn.Linear(self.hidden_size, self.num_classes)

        self.network = UGRNNNet(embedding, cell, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Update Gate RNN (UGRNN)"
