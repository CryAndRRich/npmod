# This script tests losses used in neural networks, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
np.random.seed(42)

import torch
import torch.nn.functional as F
torch.manual_seed(42)

# Importing the custom losses
from npmod.nn.losses import Loss, MAE, MSE, SmoothL1, MALE, MAPE, wMAPE, CE, BCE, KLDiv

# Function to compute diff
def compare(custom_loss: Loss, 
            torch_loss_fn: callable, 
            input_np: np.ndarray, 
            target_np: np.ndarray, 
            **kwargs) -> None:
    # Custom loss
    loss_npmod = custom_loss.forward(input_np, target_np, **kwargs) if kwargs else custom_loss.forward(input_np, target_np)
    grad_npmod = custom_loss.backward()

    # PyTorch loss
    input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)
    target_torch = torch.tensor(target_np, dtype=torch.float32)
    
    if callable(torch_loss_fn):
        loss_torch = torch_loss_fn(input_torch, target_torch, **kwargs) if kwargs else torch_loss_fn(input_torch, target_torch)
    else:
        loss_torch = torch_loss_fn(input_torch, target_torch)
    
    loss_torch.backward()
    grad_torch = input_torch.grad.detach().numpy()

    loss_diff = np.abs(loss_npmod - loss_torch.item())
    grad_diff = np.max(np.abs(grad_npmod - grad_torch))
    
    print(f"===== {custom_loss} =====")
    print(f"Loss diff: {loss_diff:.6f}")
    print(f"Grad diff: {grad_diff:.6f}")
# ====================


if __name__ == "__main__":
    N, C = 5, 3
    eps = 1e-9

    # Sample inputs
    input_reg = np.random.rand(N, 1) * 10 + 1  # > 0 for MALE
    target_reg = np.random.rand(N, 1) * 10 + 1

    input_cls = np.random.rand(N, C)
    target_cls = np.random.rand(N, C)
    target_cls = target_cls / target_cls.sum(axis=1, keepdims=True)  # normalize

    input_bce = np.random.rand(N, 1)
    target_bce = np.random.randint(0, 2, size=(N, 1))

    # MAE
    compare(MAE(), F.l1_loss, input_reg, target_reg)
    # ====================

    # MSE
    compare(MSE(), F.mse_loss, input_reg, target_reg)
    # ====================

    # SmoothL1
    compare(SmoothL1(), F.smooth_l1_loss, input_reg, target_reg, beta=1.0)
    # ====================

    # MALE
    male_npmod = MALE()
    input_torch = torch.tensor(input_reg, dtype=torch.float32, requires_grad=True)
    target_torch = torch.tensor(target_reg, dtype=torch.float32)
    loss_torch = F.mse_loss(torch.log(input_torch), torch.log(target_torch))
    loss_torch.backward()
    grad_torch = input_torch.grad.detach().numpy()
    loss_npmod = male_npmod.forward(input_reg, target_reg)
    grad_npmod = male_npmod.backward()
    print(f"===== {male_npmod} =====")
    print(f"Loss diff: {np.abs(loss_npmod - loss_torch.item()):.6f}")
    print(f"Grad diff: {np.max(np.abs(grad_npmod - grad_torch)):.6f}")
    # ====================

    # MAPE
    compare(MAPE(), lambda x, y: torch.mean(torch.abs(x - y) / (y + eps)), input_reg, target_reg)
    # ====================

    # wMAPE
    weights = np.random.rand(N, 1)
    compare(wMAPE(), lambda x, y, weights=None: torch.sum(torch.tensor(weights) * torch.abs(x - y)) / (torch.sum(torch.tensor(weights) * torch.abs(y)) + eps), 
            input_reg, target_reg, weights=weights)
    # ====================

    # CE
    target_cls_index = np.argmax(target_cls, axis=1)
    compare(CE(), lambda x, _: F.cross_entropy(x, torch.tensor(target_cls_index, dtype=torch.long)), input_cls, target_cls)
    # ====================

    # BCE
    compare(BCE(), F.binary_cross_entropy, input_bce, target_bce)
    # ====================

    # KLDiv
    input_torch = torch.tensor(input_cls, dtype=torch.float32, requires_grad=True)
    target_torch = torch.tensor(target_cls, dtype=torch.float32)
    kld_npmod = KLDiv()
    loss_npmod = kld_npmod.forward(input_cls, target_cls)
    grad_npmod = kld_npmod.backward()
    loss_torch = F.kl_div(input_torch, target_torch, reduction='batchmean')
    loss_torch.backward()
    grad_torch = input_torch.grad.detach().numpy()
    print(f"===== {kld_npmod} =====")
    print(f"Loss diff: {np.abs(loss_npmod - loss_torch.item()):.6f}")
    print(f"Grad diff: {np.max(np.abs(grad_npmod - grad_torch)):.6f}")
    # ====================

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

    ===== Mean Absolute Log Error (MALE) =====
    Loss diff: 0.000000
    Grad diff: 0.000000

    ===== Mean Absolute Percentage Error (MAPE) =====
    Loss diff: 0.000000
    Grad diff: 0.000000

    ===== Weighted Mean Absolute Percentage Error (wMAPE) =====
    Loss diff: 0.000000
    Grad diff: 0.000000

    ===== Cross Entropy Loss (CE) =====
    Loss diff: 0.106837
    Grad diff: 0.107090

    ===== Binary Cross Entropy (BCE) =====
    Loss diff: 0.000000
    Grad diff: 0.000000

    ===== Kullback-Leibler Divergence Loss (KLDiv) =====
    Loss diff: 1.686144
    Grad diff: 0.000000
    """