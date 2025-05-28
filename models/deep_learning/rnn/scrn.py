from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from ..rnn import RecurNet

class SCRNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 context_size: int, 
                 alpha: float = 0.95) -> None:
        """
        Initialize the SCRN cell

        Parameters:
            input_size: Size of the input vector at each time step
            hidden_size: Size of the hidden state h_t
            context_size: Size of the context state c_t
            alpha: Smoothing factor for context update (0 < alpha < 1)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.alpha = alpha

        self.W_x = nn.Linear(input_size, context_size)
        self.U_c = nn.Linear(context_size, hidden_size, bias=False)
        self.V_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor, 
                c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step of SCRN

        Parameters:
            x_t: Input tensor at current time step
            h_prev: Previous hidden state
            c_prev: Previous context state

        Returns:
            h_t: Current hidden state
            c_t: Current context state
        """
        c_t = (1 - self.alpha) * c_prev + self.alpha * self.W_x(x_t)
        h_t = torch.tanh(self.U_c(c_t) + self.V_h(h_prev))
        return h_t, c_t

class SCRNNet(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding, 
                 cell: SCRNCell, 
                 fc: nn.Linear) -> None:
        """
        Initialize the SCRN classification network

        Parameters:
            embedding: Embedding layer for input tokens
            cell: SCRN recurrent cell
            fc: Fully connected layer for final classification
        """
        super().__init__()
        self.embedding = embedding
        self.cell = cell
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SCRN network

        Parameters:
            x: Input tensor 

        Returns:
            out: Logits tensor 
        """
        embedded = self.embedding(x)
        batch_size, seq_len, _ = embedded.shape

        h = torch.zeros(batch_size, self.cell.hidden_size, device=embedded.device)
        c = torch.zeros(batch_size, self.cell.context_size, device=embedded.device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h, c = self.cell(x_t, h, c)

        out = self.fc(h)
        return out


class SCRN(RecurNet):
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 vocab_size: int,
                 embed_dim: int = 100,
                 hidden_size: int = 128,
                 context_size: int = 32,
                 alpha: float = 0.95,
                 num_classes: int = 10,
                 batch_size: Optional[int] = None) -> None:
        """
        Initialize the SCRN training model

        Parameters:
            learn_rate: Learning rate for optimizer
            number_of_epochs: Number of training epochs
            vocab_size: Size of the input vocabulary
            embed_dim: Size of the word embedding vectors
            hidden_size: Number of hidden units in SCRN cell
            context_size: Size of the context vector
            alpha: Smoothing factor for context update
            num_classes: Number of output classes
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
        self.context_size = context_size
        self.alpha = alpha

    def init_network(self) -> None:
        """
        Initialize SCRNNet model, weights, loss function, and optimizer
        """
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        cell = SCRNCell(input_size=self.embed_dim,
                        hidden_size=self.hidden_size,
                        context_size=self.context_size,
                        alpha=self.alpha)
        fc = nn.Linear(self.hidden_size, self.num_classes)

        self.network = SCRNNet(embedding, cell, fc)
        self.network.apply(self.init_weights)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self) -> str:
        return "Structurally Constrained Recurrent Network (SCRN)"
