# This script tests activations used in neural networks, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
np.random.seed(0)

import torch
import torch.nn as nn
torch.manual_seed(0)

# Importing the custom activations
from npmod.nn.layers import Layer
from npmod.nn.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax

# Function to compare two arrays with a tolerance
def compare_arrays(a: np.ndarray, 
                   b: np.ndarray, 
                   name: str, 
                   tol: float = 1e-5) -> bool:
    diff = np.max(np.abs(a - b))
    print(f"{name} max diff: {diff:.6f}")
    return diff < tol
# ====================

# Function to test activation layers
def test_activation(act_class: Layer, 
                    torch_act: nn.Module, 
                    alpha: float = 0.01) -> None:

    # Random input
    x_np = np.random.randn(4, 5).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)
    name = act_class()

    # Initialize activation
    if name == "LeakyReLU":
        act_np = act_class(alpha)
        act_torch = torch_act(negative_slope=alpha)
    elif name == "Softmax":
        act_np = act_class()
        act_torch = torch_act(dim=-1)  # fix dim
    else:
        act_np = act_class()
        act_torch = torch_act()

    print(f"===== Testing {name} =====")

    # Forward
    y_np = act_np.forward(x_np)
    y_torch = act_torch(x_torch)
    compare_arrays(y_np, y_torch.detach().numpy(), "Forward")

    # Backward
    grad_out = np.random.randn(*y_np.shape).astype(np.float32)
    grad_out_torch = torch.tensor(grad_out)

    grad_in_np = act_np.backward(grad_out)
    y_torch.backward(grad_out_torch)
    grad_in_torch = x_torch.grad

    compare_arrays(grad_in_np, grad_in_torch.detach().numpy(), "Backward")
# ===================


if __name__ == "__main__":
    test_activation(ReLU, nn.ReLU)
    test_activation(LeakyReLU, nn.LeakyReLU)
    test_activation(Sigmoid, nn.Sigmoid)
    test_activation(Tanh, nn.Tanh)
    test_activation(Softmax, nn.Softmax)

    """
    ===== Testing ReLU =====
    Forward max diff: 0.000000
    Backward max diff: 0.000000

    ===== Testing LeakyReLU =====
    Forward max diff: 0.000000
    Backward max diff: 0.000000

    ===== Testing Sigmoid =====
    Forward max diff: 0.000000
    Backward max diff: 0.000000

    ===== Testing Tanh =====
    Forward max diff: 0.000000
    Backward max diff: 0.000000

    ===== Testing Softmax =====
    Forward max diff: 0.000000
    Backward max diff: 0.000000
    """
