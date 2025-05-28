import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class RANCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize the RAN cell

        Parameters:
            input_size: Size of the input vector at each time step
            hidden_size: Number of hidden units
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wi = nn.Linear(input_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, 
                x: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one time step of the RAN cell

        Parameters:
            x: Input tensor at current time step
            h_prev: Previous hidden state

        Returns:
            h_t: Current hidden state
        """
        f_t = torch.sigmoid(self.Wf(x) + self.Uf(h_prev))
        i_t = torch.sigmoid(self.Wi(x) + self.Ui(h_prev))
        h_t = f_t * h_prev + i_t * x
        return h_t

class RANNet(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_size: int,
                 num_classes: int) -> None:
        """
        Initialize the RAN classification network

        Parameters:
            vocab_size: Size of the input vocabulary
            embed_dim: Dimension of the embedding vectors
            hidden_size: Number of hidden units in the RAN cell
            num_classes: Number of output classes
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.ran_cell = RANCell(embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RAN network

        Parameters:
            x: Input tensor 

        Returns:
            out: Logits tensor 
        """
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.ran_cell.hidden_size, device=x.device)

        for t in range(seq_len):
            h = self.ran_cell(x[:, t, :], h)

        out = self.fc(h)
        return out

class RAN(RecurNet):
    def init_network(self) -> None:
        """
        Initialize RANNet model, weights, loss function, and optimizer
        """
        self.network = RANNet(self.vocab_size,
                              self.embed_dim,
                              self.hidden_size,
                              self.num_classes)
        self.network.apply(self.init_weights)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Recurrent Additive Network (RAN)"
