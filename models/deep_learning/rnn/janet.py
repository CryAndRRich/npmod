import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class JANETCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize the JANET cell

        Parameters:
            input_size: Dimension of the input vector
            hidden_size: Number of hidden units in the cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate cell state
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor, 
                c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for a single time step

        Parameters:
            x_t: Input tensor at time step t 
            h_prev: Previous hidden state 
            c_prev: Previous cell state 

        Returns:
            h_t: Current hidden state 
            c_t: Current cell state 
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        f_t = torch.sigmoid(self.W_f(combined))
        c_tilde = torch.tanh(self.W_c(combined))
        c_t = f_t * c_prev + (1 - f_t) * c_tilde
        h_t = torch.tanh(c_t)
        return h_t, c_t

class JANETNet(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding, 
                 cell: JANETCell, 
                 fc: nn.Linear) -> None:
        """
        Initialize the JANET classification network

        Parameters:
            embedding: Embedding layer to map input indices to vectors
            cell: A custom JANET recurrent cell
            fc: Fully connected output layer for classification
        """
        super().__init__()
        self.embedding = embedding
        self.cell = cell
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the JANET network

        Parameters:
            x: Input tensor 

        Returns:
            out: Logits tensor 
        """
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.shape
        h = torch.zeros(batch_size, self.cell.hidden_size, device=embedded.device)
        c = torch.zeros(batch_size, self.cell.hidden_size, device=embedded.device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h, c = self.cell(x_t, h, c)

        out = self.fc(h)
        return out

class JANET(RecurNet):
    def init_network(self) -> None:
        """
        Initialize the JANET model, optimizer, and loss function
        """
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        cell = JANETCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
        fc = nn.Linear(self.hidden_size, self.num_classes)

        self.network = JANETNet(embedding, cell, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Just Another NETwork (JANET)"
