import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class SRUCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, 3 * hidden_size)
        self.bias = nn.Parameter(torch.zeros(3 * hidden_size))

    def forward(self, 
                x_t: torch.Tensor, 
                c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SRU cell

        Parameters:
            x_t: Current input at time t 
            c_prev: Previous memory cell 

        Returns:
            h_t: Current hidden state 
            c_t: Current memory cell 
        """
        x_proj = self.W(x_t) + self.bias
        x_tilde, f_t, r_t = torch.chunk(x_proj, chunks=3, dim=1)

        f_t = torch.sigmoid(f_t)
        r_t = torch.sigmoid(r_t)
        c_t = f_t * c_prev + (1 - f_t) * x_tilde
        h_t = r_t * torch.tanh(c_t) + (1 - r_t) * x_t

        return h_t, c_t

class SRUNet(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding, 
                 cell: SRUCell, 
                 fc: nn.Linear) -> None:
        super().__init__()
        self.embedding = embedding
        self.cell = cell
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SRU-based classifier

        Parameters:
            x: Input tensor 

        Returns:
            out: Logits tensor 
        """
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.shape
        c = torch.zeros(batch_size, self.cell.hidden_size, device=embedded.device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h, c = self.cell(x_t, c)

        out = self.fc(h)
        return out

class SRU(RecurNet):
    def init_network(self) -> None:
        """
        Initializes the network, optimizer, and loss criterion
        """
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        cell = SRUCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
        fc = nn.Linear(self.hidden_size, self.num_classes)

        self.network = SRUNet(embedding, cell, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Simple Recurrent Unit (SRU)"
