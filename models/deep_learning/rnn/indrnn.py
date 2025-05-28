import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class IndRNNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize a IndRNN cell

        Parameters:
            input_size: Dimension of the input vector
            hidden_size: Number of hidden units in the cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_weights = nn.Linear(input_size, hidden_size)
        self.recurrent_weights = nn.Parameter(torch.ones(hidden_size))  # Element-wise recurrent weight
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass for a single time step

        Parameters:
            x_t: Input tensor at time step t
            h_prev: Hidden state tensor from previous time step

        Returns:
            h_t: Updated hidden state tensor
        """
        input_term = self.input_weights(x_t)
        recurrent_term = self.recurrent_weights * h_prev
        h_t = torch.relu(input_term + recurrent_term + self.bias)
        return h_t

class IndRNNNet(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding, 
                 cell: IndRNNCell, 
                 fc: nn.Linear) -> None:
        """
        Initialize the IndRNN classification network

        Parameters:
            embedding: Embedding layer to map input indices to vectors
            cell: A custom IndRNN recurrent cell
            fc: Final fully connected layer for classification
        """
        super().__init__()
        self.embedding = embedding
        self.cell = cell
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass over the sequence

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

class IndRNN(RecurNet):
    def init_network(self) -> None:
        """
        Initialize the network components, optimizer, and loss function
        """
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        cell = IndRNNCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
        fc = nn.Linear(self.hidden_size, self.num_classes)

        self.network = IndRNNNet(embedding, cell, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Independently Recurrent Neural Network (IndRNN)"
