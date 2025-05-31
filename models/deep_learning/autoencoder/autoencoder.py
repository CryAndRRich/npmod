import torch
import torch.nn as nn
import torch.optim as optim

class Reshape(nn.Module):
    def __init__(self, shape: tuple) -> None:
        """
        Reshape module to reshape tensor to a specified shape

        Parameters:
            shape: The desired output shape
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input tensor to the desired shape

        Parameters:
            x: Input tensor

        Returns:
            The reshaped tensor
        """
        return x.view(-1, *self.shape)

class Autoencoder():
    def __init__(self, 
                 learn_rate: float,
                 number_of_epochs: int) -> None:
        """
        Initializes the Autoencoder model

        Parameters:
            learn_rate: The learning rate for the optimizer
            number_of_epochs: The number of training iterations
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs

    def init_network(self) -> None:
        """
        Initialize the encoder, decoder, optimizer, and loss function
        """
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28 * 28, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=12),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=12, out_features=3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=3, out_features=12),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=12, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=28 * 28),
            Reshape(shape=(1, 28, 28)),
            nn.Sigmoid()
        )

        self.network = nn.Sequential(
            self.encoder,
            self.decoder
        )

        self.network.apply(self.init_weights)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.criterion = nn.MSELoss()

    def init_weights(self, m: nn.Module) -> None:
        """
        Initialize the model parameters using the Xavier initializer

        Parameters:
            m: The module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def fit(self, 
            features: torch.Tensor) -> None:
        """
        Train the autoencoder using the provided dataset

        Parameters:
            features: The input data to be reconstructed
        """
        self.init_network()

        self.network.train()
        for _ in range(self.number_of_epochs):
            self.optimizer.zero_grad()
            outputs = self.network(features)
            loss = self.criterion(outputs, features)
            loss.backward()
            self.optimizer.step()

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode the input features into the compressed representation

        Parameters:
            features: The input data to be encoded

        Returns:
            encoded: The compressed latent representation
        """
        self.network.eval()
        with torch.no_grad():
            encoded = self.encoder(features)
        return encoded

    def __str__(self) -> str:
        return "Autoencoder"
