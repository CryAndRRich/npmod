import torch
import torch.nn as nn
import torch.optim as optim
from ....base import Model

torch.manual_seed(42)

class SVMModule(nn.Module):
    """
    Support Vector Machine (SVM) module using various kernels (linear, RBF, polynomial, sigmoid)

    The module implements different types of SVM kernels and the forward pass logic based on the selected kernel
    """
    def __init__(self, 
                 features: torch.Tensor, 
                 kernel: str = "rbf", 
                 gamma: float = 0.1, 
                 train_gamma: bool = True, 
                 degree: int = 3, 
                 b: float = 1.0) -> None:
        """
        Initializes the SVM module by defining the kernel function and the weight layer.

        Parameters:
            features: The input feature data for training 
            kernel: The kernel function to use ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: A kernel-specific parameter 
            train_gamma: Boolean flag to allow updating gamma during training 
            degree: Degree of the polynomial kernel 
            b: Constant used in polynomial and sigmoid kernels 
        """
        super().__init__()
        self.train_data = features
        self.gamma = torch.nn.Parameter(torch.FloatTensor([gamma]), requires_grad=train_gamma)
        self.b = b

        # Select the appropriate kernel function
        if kernel == "linear":
            self.kernel = self.linear
            self.C = features.size(1)
        elif kernel == "rbf":
            self.kernel = self.rbf
            self.C = features.size(0)
        elif kernel == "poly":
            self.kernel = self.poly
            self.C = features.size(0)
            self.degree = degree
        elif kernel == "sigmoid":
            self.kernel = self.sigmoid
            self.C = features.size(0)
        else: 
            raise ValueError(f"Unsupported kernel '{kernel}'. Supported kernels are 'linear', 'rbf', 'poly', and 'sigmoid'")

        # Initialize the weight layer
        self.weight = nn.Linear(in_features=self.C, out_features=1)

    def rbf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Radial Basis Function (RBF) kernel computation.

        Parameters:
            x: Input features

        Returns:
            torch.Tensor: Computed RBF kernel matrix
        """
        y = self.train_data.repeat(x.size(0), 1, 1)
        return torch.exp(-self.gamma * ((x[:, None] - y) ** 2).sum(dim=2))
    
    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear kernel computation (identity function)

        Parameters:
            x: Input features

        Returns:
            torch.Tensor: Computed linear kernel
        """
        return x
    
    def poly(self, x: torch.Tensor) -> torch.Tensor:
        """
        Polynomial kernel computation

        Parameters:
            x: Input features

        Returns:
            torch.Tensor: Computed polynomial kernel matrix
        """
        y = self.train_data.repeat(x.size(0), 1, 1)
        return (self.gamma * torch.bmm(x.unsqueeze(1), y.transpose(1, 2)) + self.b).pow(self.degree).squeeze(1)
    
    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid kernel computation

        Parameters:
            x: Input features

        Returns:
            torch.Tensor: Computed sigmoid kernel matrix
        """
        y = self.train_data.repeat(x.size(0), 1, 1)
        return torch.tanh(self.gamma * torch.bmm(x.unsqueeze(1), y.transpose(1, 2)) + self.b).squeeze(1)
    
    def hinge_loss(self, 
                   x: torch.Tensor, 
                   y: torch.Tensor) -> torch.Tensor:
        """
        Hinge loss function for SVM

        Parameters:
            x: Predictions
            y: Ground truth labels

        Returns:
            torch.Tensor: The hinge loss value
        """
        return torch.max(torch.zeros_like(y), 1 - y * x).mean()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the kernel and weight layers

        Parameters:
            x: Input features

        Returns:
            torch.Tensor: The final output after kernel transformation and linear weight application
        """
        y = self.kernel(x)  # Apply the kernel function
        y = self.weight(y)  # Apply the learned weights
        return y

class SVMModel(Model):
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int, 
                 kernel: str = "linear", 
                 gamma: float = 0.1) -> None:
        """
        Initializes the SVM model with learning rate, epochs, and kernel choice

        Parameters:
            learn_rate: The learning rate for the optimizer
            number_of_epochs: The number of training epochs
            kernel: The kernel function to use for the SVM model ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: The kernel parameter 
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Trains the SVM model on the input features and labels

        Parameters:
            features: The input features for training
            labels: The target labels corresponding to the input features
        """
        labels = labels.unsqueeze(1)
        self.model = SVMModule(features, self.kernel, self.gamma)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        self.model.train()
        for _ in range(self.number_of_epochs):
            predictions = self.model(features)  # Forward pass
            cost = self.model.hinge_loss(predictions, labels.unsqueeze(1))  # Compute hinge loss

            optimizer.zero_grad()  # Reset gradients
            cost.backward()  # Backpropagation
            optimizer.step()  # Update weights
        self.model.eval()  # Set the model to evaluation mode after training
    
    def predict(self, 
                test_features: torch.Tensor, 
                test_labels: torch.Tensor,
                get_accuracy: bool = True) -> torch.Tensor:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing
            test_labels: The true target labels corresponding to the test features
            get_accuracy: If True, calculates and prints the accuracy of predictions

        Returns:
            predictions: The prediction labels
        """
        test_labels = test_labels.unsqueeze(1)
        with torch.no_grad():
            predictions = (self.model(test_features)).detach().numpy()
            predictions = (predictions > predictions.mean())  # Binarize predictions
            test_labels = test_labels.detach().numpy()

            if get_accuracy:
                accuracy, f1 = self.evaluate(predictions, test_labels)
                print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                    self.number_of_epochs, self.number_of_epochs, accuracy, f1))
        
        return predictions
    
    def __str__(self) -> str:
        """
        Returns a string representation of the SVM model, including the chosen kernel
        """
        return "Support Vector Machine (SVM) - Kernel:'{}'".format(self.kernel)
