import numpy as np
from ..losses import Loss

class HingeEmbeddingLoss(Loss):
    def forward(self,
                input: np.ndarray,
                target: np.ndarray,
                margin: float = 1.0) -> float:
        super().forward(input, target)
        self.margin = margin

        # Ensure target has same shape as input for broadcasting
        if target.shape != input.shape:
            target = np.broadcast_to(target.reshape(-1, 1), input.shape)

        self.target = target
        self.input = input

        loss = np.mean(np.where(target == 1, input, np.maximum(0, margin - input)))
        return float(loss)

    def backward(self) -> np.ndarray:
        grad = np.where(
            self.target == 1,
            1.0,
            np.where(self.input < self.margin, -1.0, 0.0)
        )
        self.gradient = grad / self.input.shape[0]
        return self.gradient

    def __str__(self):
        return "Hinge Embedding Loss"

class MarginRankingLoss(Loss):
    def forward(self,
                input1: np.ndarray,
                input2: np.ndarray,
                target: np.ndarray,
                margin: float = 0.0) -> float:
        self.input1 = input1
        self.input2 = input2
        self.target = target
        self.margin = margin

        # Ensure shapes are consistent
        assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
        assert target.shape[0] == input1.shape[0], "target must match batch size"

        loss = np.mean(np.maximum(0, -target * (input1 - input2) + margin))
        return float(loss)

    def backward(self) -> tuple[np.ndarray, np.ndarray]:
        mask = (-self.target * (self.input1 - self.input2) + self.margin) > 0
        grad_input1 = np.where(mask, -self.target, 0.0) / self.input1.shape[0]
        grad_input2 = np.where(mask, self.target, 0.0) / self.input2.shape[0]
        return grad_input1, grad_input2

    def __str__(self):
        return "Margin Ranking Loss"
    
class TripletMarginLoss(Loss):
    def forward(self,
                anchor: np.ndarray,
                positive: np.ndarray,
                negative: np.ndarray,
                margin: float = 1.0,
                p: float = 2.0,
                eps: float = 1e-9) -> float:
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.margin = margin
        self.p = p
        self.eps = eps

        # Compute pairwise distances
        pos_dist = np.sum(np.abs(anchor - positive) ** p, axis=1) ** (1 / p)
        neg_dist = np.sum(np.abs(anchor - negative) ** p, axis=1) ** (1 / p)

        self.pos_dist = pos_dist
        self.neg_dist = neg_dist

        loss = np.mean(np.maximum(0, pos_dist - neg_dist + margin))
        return float(loss)

    def backward(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = self.anchor.shape[0]
        mask = (self.pos_dist - self.neg_dist + self.margin) > 0

        grad_anchor = np.zeros_like(self.anchor)
        grad_positive = np.zeros_like(self.positive)
        grad_negative = np.zeros_like(self.negative)

        if np.any(mask):
            pos_grad = (self.anchor - self.positive) / (self.pos_dist[:, None] + self.eps)
            neg_grad = (self.anchor - self.negative) / (self.neg_dist[:, None] + self.eps)

            grad_anchor[mask] = (pos_grad[mask] - neg_grad[mask])
            grad_positive[mask] = -pos_grad[mask]
            grad_negative[mask] = neg_grad[mask]

        return grad_anchor / N, grad_positive / N, grad_negative / N

    def __str__(self):
        return "Triplet Margin Loss"