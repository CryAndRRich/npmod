import numpy as np

class Loss():
    """
    Base class for loss functions
    """
    def __init__(self) -> None:
        self.input = None
        self.target = None

    def forward(self, 
                input: np.ndarray, 
                target: np.ndarray) -> float:
        """
        Forward pass of the loss function

        Parameters:
            input: Predictions of Layer/Model
            target: Targets to be evaluated against

        Returns:
            float: A measure of prediction accuracy
        """
        self.input = input
        self.target = target
        pass

    def backward(self) -> np.ndarray:
        """
        Backward pass of the loss function

        Returns:
            np.ndarray: Compute gradient of the loss function with respect to input
        """
        pass

    def __call__(self, input, target) -> float:
        return self.forward(input, target)

    def __str__(self) -> str:
        pass

from .regression import MAE, MSE, MALE, RSquared, MAPE, wMAPE, SmoothL1, Huber, LogCosh, QuantileLoss
from .classification import CE, BCE, FocalLoss, LabelSmoothingCE, DiceLoss
from .divergence import KLDiv, JSDiv, Wasserstein
from .ranking import HingeEmbeddingLoss, MarginRankingLoss, TripletMarginLoss