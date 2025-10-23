# This script tests rnn layers used in neural networks, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
np.random.seed(42)

import torch
import torch.nn as nn
torch.manual_seed(42)

# Importing the custom layers
from npmod.nn.rnn import RNN as npRNN
from npmod.nn.rnn import LSTM as npLSTM
from npmod.nn.rnn import GRU as npGRU

# Function to compare two arrays with a tolerance
def compare_arrays(a: np.ndarray, 
                   b: np.ndarray, 
                   name: str, 
                   tol: float = 1e-5) -> bool:
    diff = np.max(np.abs(a - b))
    print(f"{name} max diff: {diff:.6f}")
    return diff < tol
# ====================

# Function to test RNN layer
def test_rnn(batch: int = 2, 
             seq_len: int = 4, 
             input_size: int = 3, 
             hidden_size: int = 5) -> None:
    
    print("===== Testing RNN =====")

    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")
    x_torch = torch.randn(batch, seq_len, input_size, requires_grad=True)
    out_torch, h_last_torch = torch_rnn(x_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()

    npmod_rnn = npRNN(input_size, hidden_size)

    # Copy weights and biases
    with torch.no_grad():
        npmod_rnn.cell.W_x[:] = torch_rnn.weight_ih_l0.detach().numpy()
        npmod_rnn.cell.W_h[:] = torch_rnn.weight_hh_l0.detach().numpy()
        npmod_rnn.cell.b_ih[:] = torch_rnn.bias_ih_l0.detach().numpy()
        npmod_rnn.cell.b_hh[:] = torch_rnn.bias_hh_l0.detach().numpy()

    # Forward
    x_np = x_torch.detach().numpy().astype(np.float32)
    out_np, h_last_np = npmod_rnn.forward(x_np)

    compare_arrays(out_np, out_torch.detach().numpy(), "RNN outputs")
    compare_arrays(h_last_np, h_last_torch.detach().numpy(), "RNN h_last")

    # Backward
    grad_out = np.ones_like(out_np, dtype=np.float32)
    grad_input_np, _ = npmod_rnn.backward(grad_out)

    grad_input_torch = x_torch.grad.detach().numpy()
    compare_arrays(grad_input_np, grad_input_torch, "RNN dInput")
# ====================

# Function to test LSTM layer
def test_lstm(batch: int = 2,
              seq_len: int = 4,
              input_size: int = 3,
              hidden_size: int = 5) -> None:

    print("===== Testing LSTM =====")

    torch_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    x_torch = torch.randn(batch, seq_len, input_size, requires_grad=True)
    out_torch, (h_last_torch, c_last_torch) = torch_lstm(x_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()

    npmod_lstm = npLSTM(input_size, hidden_size)

    # Copy weights and biases
    with torch.no_grad():
        # PyTorch order: [i, f, g, o]
        w_ih = torch_lstm.weight_ih_l0.detach().numpy() 
        w_hh = torch_lstm.weight_hh_l0.detach().numpy()  
        b_ih = torch_lstm.bias_ih_l0.detach().numpy()
        b_hh = torch_lstm.bias_hh_l0.detach().numpy()
        bias = b_ih + b_hh

        # Split gates 
        Wi, Wf, Wc, Wo = np.split(w_ih, 4, axis=0)
        Ui, Uf, Uc, Uo = np.split(w_hh, 4, axis=0)
        bi, bf, bc, bo = np.split(bias, 4)

        # Combine (h|x) as in npmod code
        npmod_lstm.cell.W_f[:] = np.concatenate([Uf, Wf], axis=1)
        npmod_lstm.cell.W_i[:] = np.concatenate([Ui, Wi], axis=1)
        npmod_lstm.cell.W_c[:] = np.concatenate([Uc, Wc], axis=1)
        npmod_lstm.cell.W_o[:] = np.concatenate([Uo, Wo], axis=1)
        npmod_lstm.cell.b_f[:] = bf
        npmod_lstm.cell.b_i[:] = bi
        npmod_lstm.cell.b_c[:] = bc
        npmod_lstm.cell.b_o[:] = bo

    # Forward
    x_np = x_torch.detach().numpy().astype(np.float32)
    out_np, h_last_np, c_last_np = npmod_lstm.forward(x_np)

    compare_arrays(out_np, out_torch.detach().numpy(), "LSTM outputs")
    compare_arrays(h_last_np, h_last_torch.detach().numpy(), "LSTM h_last")
    compare_arrays(c_last_np, c_last_torch.detach().numpy(), "LSTM c_last")

    # Backward
    grad_out = np.ones_like(out_np, dtype=np.float32)
    grad_input_np, _, _ = npmod_lstm.backward(grad_out)
    grad_input_torch = x_torch.grad.detach().numpy()

    compare_arrays(grad_input_np, grad_input_torch, "LSTM dInput")
# ====================

# Function to test GRU layer
def test_gru(batch: int = 2,
             seq_len: int = 4,
             input_size: int = 3,
             hidden_size: int = 5) -> None:

    print("===== Testing GRU =====")

    torch_gru = nn.GRU(input_size, hidden_size, batch_first=True)
    x_torch = torch.randn(batch, seq_len, input_size, requires_grad=True)
    out_torch, h_last_torch = torch_gru(x_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()

    npmod_gru = npGRU(input_size, hidden_size)

    # Copy weights and biases
    with torch.no_grad():
        W_ih = torch_gru.weight_ih_l0.detach().numpy()
        W_hh = torch_gru.weight_hh_l0.detach().numpy()
        b_ih = torch_gru.bias_ih_l0.detach().numpy()  
        b_hh = torch_gru.bias_hh_l0.detach().numpy()

    # PyTorch order: [r, z, n]
    W_ih_r, W_ih_z, W_ih_n = np.split(W_ih, 3, axis=0)
    W_hh_r, W_hh_z, W_hh_n = np.split(W_hh, 3, axis=0)
    b_ih_r, b_ih_z, b_ih_n = np.split(b_ih, 3)
    b_hh_r, b_hh_z, b_hh_n = np.split(b_hh, 3)

    # Map to npmod naming [z, r, h] 
    npmod_gru.cell.W_z[:] = np.concatenate([W_hh_z, W_ih_z], axis=1)
    npmod_gru.cell.W_r[:] = np.concatenate([W_hh_r, W_ih_r], axis=1)
    npmod_gru.cell.W_h[:] = np.concatenate([W_hh_n, W_ih_n], axis=1)

    npmod_gru.cell.b_z[:] = b_ih_z + b_hh_z
    npmod_gru.cell.b_r[:] = b_ih_r + b_hh_r
    npmod_gru.cell.b_h_ih[:] = b_ih_n
    npmod_gru.cell.b_h_hh[:] = b_hh_n

    # Forward
    x_np = x_torch.detach().numpy().astype(np.float32)
    out_np, h_last_np = npmod_gru.forward(x_np)

    # Compare outputs
    compare_arrays(out_np, out_torch.detach().numpy(), "GRU outputs")
    compare_arrays(h_last_np, h_last_torch.detach().numpy(), "GRU h_last")

    # Backward
    grad_out = np.ones_like(out_np, dtype=np.float32)
    grad_input_np, _ = npmod_gru.backward(grad_out)

    grad_input_torch = x_torch.grad.detach().numpy()
    compare_arrays(grad_input_np, grad_input_torch, "GRU dInput")
# ====================

if __name__ == "__main__":
    test_rnn()
    test_lstm()
    test_gru()

    """
    ===== Testing RNN =====
    RNN outputs max diff: 0.000000
    RNN h_last max diff: 0.000000
    RNN dInput max diff: 0.000000

    ===== Testing LSTM =====     
    LSTM outputs max diff: 0.000000
    LSTM h_last max diff: 0.000000
    LSTM c_last max diff: 0.000000
    LSTM dInput max diff: 0.000000
    
    ===== Testing GRU =====       
    GRU outputs max diff: 0.000000
    GRU h_last max diff: 0.000000 
    GRU dInput max diff: 0.000000 
    """