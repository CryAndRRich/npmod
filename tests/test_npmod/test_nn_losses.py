# This script tests losses used in neural networks, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
np.random.seed(42)

import torch
import torch.nn.functional as F
torch.manual_seed(42)

# Importing the custom losses
from npmod.nn.losses import Loss
from npmod.nn.losses.regression import MAE, MSE, SmoothL1, Huber
from npmod.nn.losses.classification import CE, BCE, LabelSmoothingCE
from npmod.nn.losses.divergence import KLDiv
from npmod.nn.losses.ranking import HingeEmbeddingLoss, MarginRankingLoss, TripletMarginLoss

# Function to compute diff
def compare(custom_loss: Loss, 
            torch_loss_fn: callable, 
            *inputs, **kwargs) -> None:

    np_inputs = [np.array(x, dtype=np.float32) for x in inputs]
    torch_inputs = [torch.tensor(x, dtype=torch.float32, requires_grad=(i==0)) for i, x in enumerate(np_inputs)]

    # Custom loss
    loss_npmod = custom_loss.forward(*np_inputs, **kwargs)
    grad_npmod = custom_loss.backward()

    # PyTorch loss
    loss_torch = torch_loss_fn(*torch_inputs, **kwargs)
    loss_torch.backward()
    grad_torch = torch_inputs[0].grad.detach().numpy()
    
    loss_diff = np.abs(loss_npmod - loss_torch.item())
    grad_diff = np.max(np.abs(grad_npmod - grad_torch))
    
    print(f"===== {custom_loss} =====")
    print(f"Loss diff: {loss_diff:.6f}")
    print(f"Grad diff: {grad_diff:.6f}")
# ====================


if __name__ == "__main__":
    N, C, D = 5, 3, 4
    eps = 1e-9

    # Regression data
    input_reg = np.random.rand(N, 1) * 10 + 1
    target_reg = np.random.rand(N, 1) * 10 + 1

    # Classification data
    input_cls = np.random.rand(N, C)
    target_cls = np.random.rand(N, C)
    target_cls /= target_cls.sum(axis=1, keepdims=True)
    target_idx = np.argmax(target_cls, axis=1)

    # BCE data
    input_bce = np.random.rand(N, 1)
    target_bce = np.random.randint(0, 2, size=(N, 1))

    # Ranking data
    x1 = np.random.rand(N, D)
    x2 = np.random.rand(N, D)
    target_rank = np.random.choice([-1, 1], size=(N,))
    anchor = np.random.rand(N, D)
    positive = np.random.rand(N, D)
    negative = np.random.rand(N, D)

    x1_flat = np.random.rand(N)                      
    target_hinge = np.random.choice([-1, 1], size=(N,))

    # Comparisons
    compare(MAE(), F.l1_loss, input_reg, target_reg)
    compare(MSE(), F.mse_loss, input_reg, target_reg)
    compare(SmoothL1(), F.smooth_l1_loss, input_reg, target_reg)
    compare(Huber(), F.huber_loss, input_reg, target_reg, delta=1.0)

    compare(CE(), lambda a, b: F.cross_entropy(a, torch.tensor(target_idx, dtype=torch.long)), input_cls, target_cls)
    compare(BCE(), F.binary_cross_entropy, input_bce, target_bce)
    compare(LabelSmoothingCE(), lambda a, b: F.cross_entropy(a, torch.tensor(target_idx, dtype=torch.long), label_smoothing=0.1), input_cls, target_cls)

    compare(KLDiv(), lambda a, b: F.kl_div(a, b, reduction="batchmean"), input_cls, target_cls)

    compare(HingeEmbeddingLoss(), F.hinge_embedding_loss, x1_flat, target_hinge)
    compare(MarginRankingLoss(), F.margin_ranking_loss, x1.mean(axis=1), x2.mean(axis=1), target_rank)
    compare(TripletMarginLoss(), F.triplet_margin_loss, anchor, positive, negative)

    """
    ===== Mean Absolute Error (MAE) =====
    Loss diff: 0.000000
    Grad diff: 0.000000
    ===== Mean Squared Error (MSE) =====
    Loss diff: 0.000000
    Grad diff: 0.000000
    ===== Smooth L1 Loss (SmoothL1) =====
    Loss diff: 0.000000
    Grad diff: 0.000000
    ===== Huber Loss =====
    Loss diff: 0.000000
    Grad diff: 0.000000

    ===== Cross Entropy Loss (CE) =====
    Loss diff: 0.106837
    Grad diff: 0.107090
    ===== Binary Cross Entropy (BCE) =====
    Loss diff: 0.000000
    Grad diff: 0.000000
    ===== Label Smoothing Cross Entropy =====
    Loss diff: 0.096153
    Grad diff: 0.096381

    ===== Kullback-Leibler Divergence Loss (KLDiv) =====
    Loss diff: 1.686144
    Grad diff: 0.000000
    
    ===== Hinge Embedding Loss =====
    Loss diff: 0.000000
    Grad diff: 0.000000
    ===== Margin Ranking Loss ===== 
    Loss diff: 0.000000
    Grad diff: 0.400000
    ===== Triplet Margin Loss =====
    Loss diff: 0.000000
    Grad diff: 0.388586
    """