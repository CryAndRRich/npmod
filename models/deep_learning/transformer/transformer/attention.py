import torch
import torch.nn as nn

class ScaledDotAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int) -> None:
        """
        Scaled Dot-Product Attention

        Parameters:
            d_model: Dimension of the model
            nhead: Number of attention heads
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.nhead = nhead

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                attn_mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:

        out, _ = self.attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return out


class LinformerAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 seq_len: int = 512, 
                 k: int = 128) -> None:
        """
        Linformer Attention

        Parameters:
            d_model: Dimension of the model
            nhead: Number of attention heads
            seq_len: Length of the input sequence
            k: Dimension to project the keys and values to
        """
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.k = k
        self.seq_len = seq_len
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                attn_mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        _, L, _ = k.size()
        if L > self.seq_len:
            raise ValueError(f"Input sequence length {L} exceeds max Linformer seq_len={self.seq_len}")

        E = self.E[:L, :]
        k_proj = torch.matmul(k.transpose(1, 2), E).transpose(1, 2)
        v_proj = torch.matmul(v.transpose(1, 2), E).transpose(1, 2)

        if attn_mask is not None and attn_mask.size(-1) != k_proj.size(1):
            attn_mask = None
        
        out, _ = self.attn(q, k_proj, v_proj, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return out


class PerformerAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 nb_features: int = 256) -> None:
        """
        Performer Attention
        
        Parameters:
            d_model: Dimension of the model
            nhead: Number of attention heads
            nb_features: Number of random features for approximation
        """
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.nb_features = nb_features
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _phi(self, q: torch.Tensor) -> torch.Tensor:
        return torch.exp(-q ** 2 / 2)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                attn_mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:

        Q = self._phi(self.query_proj(q))
        K = self._phi(self.key_proj(k))
        V = self.value_proj(v)

        if key_padding_mask is not None:
            key_mask = key_padding_mask.unsqueeze(-1).expand_as(K)
            K = K.masked_fill(key_mask, 0.0)
            V = V.masked_fill(key_mask, 0.0)

        KV = torch.einsum("bnd, bnl->bdl", K, V)
        Z = 1.0 / (torch.einsum("bnd, bd->bn", Q, K.sum(dim=1)) + 1e-6)
        out = torch.einsum("bnd, bdl, bn->bnl", Q, KV, Z)
        return self.out_proj(out)


class LocalAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 window_size: int = 128) -> None:
        """
        Local Windowed Attention
        
        Parameters:
            d_model: Dimension of the model
            nhead: Number of attention heads
            window_size: Size of the local attention window
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor = None, 
                v: torch.Tensor = None, 
                attn_mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:

        _, L, _ = q.shape
        outputs = []
        for start in range(0, L, self.window_size):
            end = min(L, start + self.window_size)
            q_chunk = q[:, start:end, :]
            k_chunk = k[:, start:end, :]
            v_chunk = v[:, start:end, :]
            mask_chunk = None
            pad_chunk = None
            if attn_mask is not None:
                mask_chunk = attn_mask[start:end, start:end]
            if key_padding_mask is not None:
                pad_chunk = key_padding_mask[:, start:end]
            out, _ = self.attn(q_chunk, k_chunk, v_chunk, attn_mask=mask_chunk, key_padding_mask=pad_chunk)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class SparseAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 stride: int = 4) -> None:
        """
        Sparse Attention
        
        Parameters:
            d_model: Dimension of the model
            nhead: Number of attention heads
            stride: Stride for sampling keys and values
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.stride = stride
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                attn_mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:

        k_sparse = k[:, ::self.stride, :]
        v_sparse = v[:, ::self.stride, :]
        sparse_pad_mask = None
        if key_padding_mask is not None:
            sparse_pad_mask = key_padding_mask[:, ::self.stride]
        out, _ = self.attn(q, k_sparse, v_sparse, attn_mask=attn_mask, key_padding_mask=sparse_pad_mask)
        return out


class CosineAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 eps: float = 1e-8) -> None:
        """
        Cosine Similarity Based Attention
        
        Parameters:
            d_model: Dimension of the model
            nhead: Number of attention heads
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.nhead = nhead
        self.scale = d_model ** -0.5
        self.eps = eps
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                attn_mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:

        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)

        Q_norm = Q / (Q.norm(dim=-1, keepdim=True) + self.eps)
        K_norm = K / (K.norm(dim=-1, keepdim=True) + self.eps)
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), float('-inf'))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, :], float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return self.out_proj(out)