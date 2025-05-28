import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class MGUCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize MGU cell

        Parameters:
            input_size: Dimension of input vector
            hidden_size: Number of hidden units in the cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters for update gate z_t
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)

        # Parameters for candidate hidden state h~_t
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass for one time step

        Parameters:
            x_t: Input at current time step 
            h_prev: Previous hidden state 

        Returns:
            h_t: Updated hidden state 
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        z_t = torch.sigmoid(self.W_z(combined))
        h_tilde = torch.tanh(self.W_h(combined))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t

class MGUNet(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding, 
                 cell: MGUCell, 
                 fc: nn.Linear) -> None:
        """
        Initialize the MGU classification network

        Parameters:
            embedding: Embedding layer for input tokens
            cell: Recurrent cell (MGU)
            fc: Fully connected layer for output classification
        """
        super().__init__()
        self.embedding = embedding
        self.cell = cell
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

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

class MGU(RecurNet):
    def init_network(self) -> None:
        """
        Initialize the embedding layer, MGU cell, classification layer, 
        and the optimizer and loss function
        """
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        cell = MGUCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
        fc = nn.Linear(self.hidden_size, self.num_classes)

        self.network = MGUNet(embedding, cell, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Minimal Gated Unit (MGU)"
