# This script tests layers used in neural networks, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn

# Importing the custom layers
from npmod.nn.layers.linear import Linear as npLinear
from npmod.nn.layers.conv import Conv as npConv
from npmod.nn.layers.pooling import Pooling as npPooling
from npmod.nn.layers.batchnorm import BatchNorm as npBatchNorm
from npmod.nn.layers.embedding import Embedding as npEmbedding
from npmod.nn.layers.flatten import Flatten as npFlatten

# Function to compare two arrays with a tolerance
def compare_arrays(a: np.ndarray, 
                   b: np.ndarray, 
                   name: str, 
                   tol: float = 1e-5) -> bool:
    diff = np.max(np.abs(a - b))
    print(f"{name} max diff: {diff:.6f}")
    return diff < tol
# ====================

# Function to test Linear layer
def test_linear() -> None:
    print("===== Testing Linear =====")
    in_dim, out_dim = 10, 6
    x = np.random.randn(4, in_dim).astype(np.float32)

    npmod_layer = npLinear(in_dim, out_dim)
    torch_layer = nn.Linear(in_dim, out_dim, bias=True)

    torch_layer.weight.data = torch.tensor(npmod_layer.weight, dtype=torch.float32)
    torch_layer.bias.data = torch.tensor(npmod_layer.bias, dtype=torch.float32)

    x_torch = torch.tensor(x, requires_grad=True)

    y_npmod = npmod_layer.forward(x)
    y_torch = torch_layer(x_torch)

    compare_arrays(y_npmod, y_torch.detach().numpy(), "Forward")

    grad_out = np.random.randn(*y_npmod.shape).astype(np.float32)
    grad_in_npmod = npmod_layer.backward(grad_out)

    grads_torch = torch.autograd.grad(
        y_torch,
        [x_torch, torch_layer.weight, torch_layer.bias],
        torch.tensor(grad_out, dtype=torch.float32)
    )

    compare_arrays(grad_in_npmod, grads_torch[0].detach().numpy(), "Grad input")
    compare_arrays(npmod_layer.weight_grad, grads_torch[1].detach().numpy(), "Grad weight")
    compare_arrays(npmod_layer.bias_grad, grads_torch[2].detach().numpy(), "Grad bias")
# ====================

# Function to test Conv2D layer
def test_conv2d() -> None:
    print("===== Testing Conv2D =====")
    x = np.random.randn(2, 3, 5, 5).astype(np.float32)

    npmod_layer = npConv(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, num_dims=2)
    torch_layer = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)

    torch_layer.weight.data = torch.tensor(npmod_layer.weight, dtype=torch.float32)
    torch_layer.bias.data = torch.tensor(npmod_layer.bias, dtype=torch.float32)

    x_torch = torch.tensor(x, requires_grad=True)

    y_npmod = npmod_layer.forward(x)
    y_torch = torch_layer(x_torch)

    compare_arrays(y_npmod, y_torch.detach().numpy(), "Forward")

    grad_out = np.random.randn(*y_npmod.shape).astype(np.float32)
    grad_in_npmod = npmod_layer.backward(grad_out)

    grads_torch = torch.autograd.grad(
        y_torch,
        [x_torch, torch_layer.weight, torch_layer.bias],
        torch.tensor(grad_out, dtype=torch.float32)
    )

    compare_arrays(grad_in_npmod, grads_torch[0].detach().numpy(), "Grad input")
    compare_arrays(npmod_layer.weight_grad, grads_torch[1].detach().numpy(), "Grad weight")
    compare_arrays(npmod_layer.bias_grad, grads_torch[2].detach().numpy(), "Grad bias")
# ====================

# Function to test BatchNorm1d layer
def test_batchnorm() -> None:
    print("===== Testing BatchNorm1d =====")
    x = np.random.randn(4, 6).astype(np.float32)

    npmod_layer = npBatchNorm(num_features=6)
    torch_layer = nn.BatchNorm1d(6, affine=True, track_running_stats=False)

    torch_layer.weight.data = torch.tensor(npmod_layer.gamma, dtype=torch.float32)
    torch_layer.bias.data = torch.tensor(npmod_layer.beta, dtype=torch.float32)

    x_torch = torch.tensor(x, requires_grad=True)

    y_npmod = npmod_layer.forward(x)
    y_torch = torch_layer(x_torch)

    compare_arrays(y_npmod, y_torch.detach().numpy(), "Forward")

    grad_out = np.random.randn(*y_npmod.shape).astype(np.float32)
    grad_in_npmod = npmod_layer.backward(grad_out)

    grads_torch = torch.autograd.grad(
        y_torch,
        [x_torch, torch_layer.weight, torch_layer.bias],
        torch.tensor(grad_out, dtype=torch.float32)
    )

    compare_arrays(grad_in_npmod, grads_torch[0].detach().numpy(), "Grad input")
    compare_arrays(npmod_layer.gamma_grad, grads_torch[1].detach().numpy(), "Grad gamma")
    compare_arrays(npmod_layer.beta_grad, grads_torch[2].detach().numpy(), "Grad beta")
# ====================

# Function to test Embedding layer
def test_embedding() -> None:
    print("===== Testing Embedding =====")
    x = np.array([[1, 3, 4], [0, 2, 1]], dtype=np.int64)

    npmod_layer = npEmbedding(num_embeddings=5, embedding_dim=4)
    torch_layer = nn.Embedding(5, 4)

    torch_layer.weight.data = torch.tensor(npmod_layer.weight, dtype=torch.float32)

    x_torch = torch.tensor(x, dtype=torch.long)

    y_npmod = npmod_layer.forward(x)
    y_torch = torch_layer(x_torch)

    compare_arrays(y_npmod, y_torch.detach().numpy(), "Forward")

    grad_out = np.random.randn(*y_npmod.shape).astype(np.float32)
    npmod_layer.backward(grad_out)

    y_torch.backward(torch.tensor(grad_out, dtype=torch.float32))
    compare_arrays(npmod_layer.grad_weight, torch_layer.weight.grad.detach().numpy(), "Grad weight")
# ====================

# Function to test Flatten layer
def test_flatten() -> None:
    print("===== Testing Flatten =====")
    x = np.random.randn(2, 3, 4).astype(np.float32)

    npmod_layer = npFlatten()
    torch_layer = nn.Flatten()

    x_torch = torch.tensor(x, requires_grad=True)

    y_npmod = npmod_layer.forward(x)
    y_torch = torch_layer(x_torch)

    compare_arrays(y_npmod, y_torch.detach().numpy(), "Forward")

    grad_out = np.random.randn(*y_npmod.shape).astype(np.float32)
    grad_in_npmod = npmod_layer.backward(grad_out)

    grad_in_torch = torch.autograd.grad(
        y_torch,
        x_torch,
        torch.tensor(grad_out, dtype=torch.float32)
    )[0]

    compare_arrays(grad_in_npmod, grad_in_torch.detach().numpy(), "Grad input")
# ====================

# Function to test Pooling layer
def test_pooling() -> None:
    print("===== Testing MaxPool2d =====")
    x = np.random.randn(1, 1, 4, 4).astype(np.float32)

    npmod_layer = npPooling(kernel_size=2, stride=2, num_dims=2, mode="max")
    torch_layer = nn.MaxPool2d(2, 2)

    x_torch = torch.tensor(x, requires_grad=True)

    y_npmod = npmod_layer.forward(x)
    y_torch = torch_layer(x_torch)

    compare_arrays(y_npmod[0, 0], y_torch.detach().numpy()[0, 0], "Forward")

    grad_out = np.random.randn(*y_npmod.shape).astype(np.float32)
    grad_in_npmod = npmod_layer.backward(grad_out)

    grad_in_torch = torch.autograd.grad(
        y_torch,
        x_torch,
        torch.tensor(grad_out, dtype=torch.float32)
    )[0]

    compare_arrays(grad_in_npmod[0, 0], grad_in_torch.detach().numpy()[0, 0], "Grad input")
# ====================


if __name__ == "__main__":
    test_linear()
    """
    ===== Testing Linear =====
    Forward max diff: 0.000000
    Grad input max diff: 0.000000
    Grad weight max diff: 0.000000
    Grad bias max diff: 0.000000
    """

    test_conv2d()
    """
    ===== Testing Conv2D =====
    Forward max diff: 0.000000
    Grad input max diff: 0.000000
    Grad weight max diff: 0.000002
    Grad bias max diff: 0.000001
    """

    test_batchnorm()
    """
    ===== Testing BatchNorm1d =====
    Forward max diff: 0.000000
    Grad input max diff: 0.000000
    Grad gamma max diff: 0.000000
    Grad beta max diff: 0.000000
    """

    test_embedding()
    """
    ===== Testing Embedding =====
    Forward max diff: 0.000000
    Grad weight max diff: 0.000000
    """

    test_flatten()
    """===== Testing Flatten =====
    Forward max diff: 0.000000
    Grad input max diff: 0.000000
    """

    test_pooling()
    """
    ===== Testing MaxPool2d =====
    Forward max diff: 0.000000
    Grad input max diff: 0.000000
    """