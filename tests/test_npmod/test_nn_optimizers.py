# This script tests optimizers used in neural networks, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
np.random.seed(42)

import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)

# Importing the custom optimizers
from npmod.nn.layers import Layer
from npmod.nn.layers import Linear as npLinear
from npmod.nn.layers import Conv as npConv
from npmod.nn.optimizers import Optimizer
from npmod.nn.optimizers.classical import GD, SGD
from npmod.nn.optimizers.adaptive import AdaGrad, RMSprop, Adam, AdamW, RAdam

# Function to compare two arrays with a tolerance
def compare_arrays(a: np.ndarray, 
                   b: np.ndarray, 
                   name: str, 
                   tol: float = 1e-5) -> bool:
    diff = np.max(np.abs(a - b))
    print(f"{name} max diff: {diff:.6f}")
    return diff < tol
# ====================

# Function to test optimizers with Linear layer
def test_optimizer_linear(npLinear: Layer, 
                          npOptimizerClass: Optimizer, 
                          torchOptimizerClass: optim.Optimizer, 
                          in_dim: int = 5, 
                          out_dim: int = 3, 
                          lr: float = 0.01, 
                          steps: int = 3) -> None:

    print(f"===== Testing Linear + {npOptimizerClass.__name__} =====")

    # Random input
    x = np.random.randn(4, in_dim).astype(np.float32)
    y_true = np.random.randn(4, out_dim).astype(np.float32)

    # Initialize layers
    np_layer = npLinear(in_dim, out_dim)
    torch_layer = nn.Linear(in_dim, out_dim, bias=True)

    torch_layer.weight.data = torch.tensor(np_layer.weight, dtype=torch.float32)
    torch_layer.bias.data = torch.tensor(np_layer.bias, dtype=torch.float32)

    # Optimizers
    np_opt = npOptimizerClass([np_layer], learn_rate=lr)
    torch_opt = torchOptimizerClass(torch_layer.parameters(), lr=lr)

    # Loss function
    loss_fn = nn.MSELoss()

    for step in range(steps):
        y_pred_npmod = np_layer.forward(x)
        loss_npmod = np.mean((y_pred_npmod - y_true) ** 2)

        grad_out = 2 * (y_pred_npmod - y_true) / (y_true.shape[0] * y_true.shape[1])
        _ = np_layer.backward(grad_out)

        np_opt.step()

        x_torch = torch.tensor(x, requires_grad=True)
        y_true_torch = torch.tensor(y_true)

        y_pred_torch = torch_layer(x_torch)
        loss_torch = loss_fn(y_pred_torch, y_true_torch)

        torch_opt.zero_grad()
        loss_torch.backward()
        torch_opt.step()

        print(f"Step {step + 1}:")
        compare_arrays(np_layer.weight, torch_layer.weight.detach().numpy(), "- Weight")
        compare_arrays(np_layer.bias, torch_layer.bias.detach().numpy(), "- Bias")
        compare_arrays(loss_npmod, loss_torch.item(), "- Loss")
# ====================

# Function to test optimizers with Conv2D layer
def test_optimizer_conv2d(npConv: Layer,
                          npOptimizerClass: Optimizer,
                          torchOptimizerClass: optim.Optimizer,
                          in_channels: int = 3,
                          out_channels: int = 4,
                          kernel_size: int = 3,
                          stride: int = 1,
                          padding: int = 1,
                          lr: float = 0.01,
                          steps: int = 3) -> None:
    
    print(f"===== Testing Conv2D + {npOptimizerClass.__name__} =====")

    # Random input and target
    x = np.random.randn(2, in_channels, 5, 5).astype(np.float32)
    y_true = np.random.randn(2, out_channels, 5, 5).astype(np.float32)

    # Initialize layers
    np_layer = npConv(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      num_dims=2)
    torch_layer = nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)

    # Sync parameters
    torch_layer.weight.data = torch.tensor(np_layer.weight, dtype=torch.float32)
    torch_layer.bias.data = torch.tensor(np_layer.bias, dtype=torch.float32)

    # Optimizers
    np_opt = npOptimizerClass([np_layer], learn_rate=lr)
    torch_opt = torchOptimizerClass(torch_layer.parameters(), lr=lr)

    # Loss
    loss_fn = nn.MSELoss()

    for step in range(steps):
        # numpy forward + loss
        y_pred_np = np_layer.forward(x)
        loss_np = np.mean((y_pred_np - y_true) ** 2)

        # numpy backward
        grad_out = 2 * (y_pred_np - y_true) / (y_true.shape[0] * np.prod(y_true.shape[1:]))
        _ = np_layer.backward(grad_out)
        np_opt.step()

        # torch forward + loss
        x_torch = torch.tensor(x, requires_grad=True)
        y_true_torch = torch.tensor(y_true)

        y_pred_torch = torch_layer(x_torch)
        loss_torch = loss_fn(y_pred_torch, y_true_torch)

        torch_opt.zero_grad()
        loss_torch.backward()
        torch_opt.step()

        # compare
        print(f"Step {step + 1}:")
        compare_arrays(np_layer.weight, torch_layer.weight.detach().numpy(), "- Weight")
        compare_arrays(np_layer.bias, torch_layer.bias.detach().numpy(), "- Bias")
        compare_arrays(loss_np, loss_torch.item(), "- Loss")
# ====================


if __name__ == "__main__":
    # Testing optimizers with Linear layer
    #test_optimizer_linear(npLinear, GD, lambda p, lr: optim.SGD(p, lr=lr, momentum=0))
    #test_optimizer_linear(npLinear, SGD, lambda p, lr: optim.SGD(p, lr=lr, momentum=0.01, dampening=0, nesterov=False))
    
    #test_optimizer_linear(npLinear, AdaGrad, optim.Adagrad)
    #test_optimizer_linear(npLinear, RMSprop, lambda p, lr: optim.RMSprop(p, lr=lr, alpha=0.9, eps=1e-8))
    #test_optimizer_linear(npLinear, Adam, optim.Adam)
    #test_optimizer_linear(npLinear, AdamW, optim.AdamW)
    #test_optimizer_linear(npLinear, RAdam, optim.RAdam)
    """
    ===== Testing Linear + GD =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Linear + SGD =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Linear + AdaGrad =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Linear + RMSprop =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Linear + Adam =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    ===== Testing Linear + AdamW =====
    Step 1:
    - Weight max diff: 0.000001
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000008
    - Bias max diff: 0.000001
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000055
    - Bias max diff: 0.000003
    - Loss max diff: 0.000003

    ===== Testing Linear + RAdam =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    """
    # ====================

    # Testing optimizers with Conv2D layer
    test_optimizer_conv2d(npConv, GD, optim.SGD)
    test_optimizer_conv2d(npConv, SGD, lambda p, lr: optim.SGD(p, lr=lr, momentum=0.01))

    test_optimizer_conv2d(npConv, AdaGrad, optim.Adagrad)
    test_optimizer_conv2d(npConv, RMSprop, lambda p, lr: optim.RMSprop(p, lr=lr, alpha=0.9, eps=1e-8))
    test_optimizer_conv2d(npConv, Adam, optim.Adam)
    test_optimizer_conv2d(npConv, AdamW, optim.AdamW)
    test_optimizer_conv2d(npConv, RAdam, optim.RAdam)

    """
    ===== Testing Conv2D + GD =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Conv2D + SGD =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Conv2D + AdaGrad =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Conv2D + RMSprop =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Conv2D + Adam =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000

    ===== Testing Conv2D + AdamW =====
    Step 1:
    - Weight max diff: 0.000003
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.001492
    - Bias max diff: 0.000005
    - Loss max diff: 0.000001
    Step 3:
    - Weight max diff: 0.002589
    - Bias max diff: 0.000024
    - Loss max diff: 0.000003
    
    ===== Testing Conv2D + RAdam =====
    Step 1:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 2:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    Step 3:
    - Weight max diff: 0.000000
    - Bias max diff: 0.000000
    - Loss max diff: 0.000000
    """
    # ====================
