import torch
import torch.nn as nn
import torch.optim as optim
from base_model import ModelML

torch.manual_seed(42)

class LogisticRegressionModule(nn.Module):
    """
    Logistic regression model using PyTorch's nn.Module
    """
    def __init__(self, n: int):
        """
        Initializes the logistic regression model by defining a single linear layer
        and a sigmoid activation function

        Parameters:
        n: Number of input features
        """
        super().__init__()
        self.linear = nn.Linear(in_features=n, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer and sigmoid activation

        Parameters:
        x: Input features

        --------------------------------------------------
        Returns:
        Tensor: Output of the sigmoid-activated linear layer
        """
        return self.sigmoid(self.linear(x))

class LogisticRegressionPytorch(ModelML):
    def __init__(self, 
                 learn_rate: float, 
                 number_of_epochs: int):
        """
        Initializes the Logistic Regression model with the learning rate and number of epochs

        Parameters:
        learn_rate: The learning rate for the optimizer
        number_of_epochs: The number of training iterations to run
        """
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Trains the logistic regression model on the input data

        Parameters:
        features: The input features for training 
        labels: The target labels corresponding to the input features 
        """
        labels = labels.to(dtype=torch.float)
        _, n = features.shape

        self.model = LogisticRegressionModule(n)  # Initialize the model
        self.criterion = nn.BCELoss()  # Binary cross-entropy loss
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        self.model.train()  # Set the model to training mode
        for _ in range(self.number_of_epochs):
            prediction = self.model(features)  # Forward pass through the model
            
            cost = self.criterion(prediction, labels)  # Compute the binary cross-entropy loss
            
            optimizer.zero_grad()  # Reset gradients
            cost.backward()  # Backpropagation
            optimizer.step()  # Update weights
        
        self.model.eval()  # Set the model to evaluation mode
    
    def predict(self, 
                test_features: torch.Tensor, 
                test_labels: torch.Tensor) -> None:
        """
        Predicts the labels for the test data using the trained Logistic Regression model

        Parameters:
        test_features: The input features for testing 
        test_labels: The target labels corresponding to the test features 
        """
        test_labels = test_labels.to(dtype=torch.float)
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(test_features)  # Get the model's output
            predictions = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
            predictions = predictions.detach().numpy()
            test_labels = test_labels.detach().numpy()

            # Evaluate accuracy and F1 score
            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                   self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self) -> str:
        return "Logistic Regression (Pytorch)"
