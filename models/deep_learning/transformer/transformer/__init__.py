from .positional_encoding import *
from .normalization import *
from .attention import *
from .feed_forward import *

from .learn_rate import *

def get_positional_encoding(pe_type: str,
                            d_model: int,
                            max_len: int = 5000,
                            num_heads: int = None,
                            max_distance: int = 128) -> nn.Module:
    pe_type = pe_type.lower()
    
    if pe_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
    
    elif pe_type == "learned":
        return LearnedPositionalEncoding(d_model=d_model,max_len=max_len)
    
    elif pe_type in ["relative_bias", "relative"]:
        if num_heads is None:
            raise ValueError("'num_heads' must be provided for RelativePositionalBias")
        return RelativePositionalBias(num_heads=num_heads, max_distance=max_distance)
    
    elif pe_type == "rotary":
        return RotaryPositionalEmbedding(dim=d_model)
    
    elif pe_type in ["alibi", "alibias"]:
        if num_heads is None:
            raise ValueError("num_heads must be provided for ALiBiBias")
        return ALiBiBias(num_heads=num_heads)
    
    else:
        raise ValueError(f"Positional encoding type '{pe_type}' not supported")
    

def get_norm(norm_type: str, 
             d_model: int) -> nn.Module:
    norm_type = norm_type.lower()

    if norm_type == "none":
        return nn.Identity()

    elif norm_type in ["layer", "layernorm"]:
        return LayerNorm(d_model=d_model)
    
    elif norm_type in ["rms", "rmsnorm"]:
        return RMSNorm(d_model=d_model)
    
    elif norm_type in ["scale", "scalenorm"]:
        return ScaleNorm(d_model=d_model)
    
    elif norm_type in ["ada", "adanorm"]:
        return AdaNorm(d_model=d_model)
    
    else:
        raise ValueError(f"Normalization type '{norm_type}' not supported")
    

def get_attention(attn_type: str,
                  d_model: int,
                  nhead: int,
                  seq_len: int,
                  k: int = 128,
                  nb_features: int = 256,
                  window_size: int = 128,
                  stride: int = 4,
                  eps: float = 1e-8) -> nn.Module:
    attn_type = attn_type.lower()
    
    if attn_type in ["scaled_dot", "scaled_dot_product", "vanilla"]:
        return ScaledDotAttention(d_model=d_model, nhead=nhead)
    
    elif attn_type == "linformer":
        return LinformerAttention(d_model=d_model, nhead=nhead, seq_len=seq_len, k=k)
    
    elif attn_type == "performer":
        return PerformerAttention(d_model=d_model, nhead=nhead, nb_features=nb_features)
    
    elif attn_type == "local":
        return LocalAttention(d_model=d_model, nhead=nhead, window_size=window_size)
    
    elif attn_type == "sparse":
        return SparseAttention(d_model=d_model, nhead=nhead, stride=stride)
    
    elif attn_type == "cosine":
        return CosineAttention(d_model=d_model, nhead=nhead, eps=eps)
    
    else:
        raise ValueError(f"Attention type '{attn_type}' not supported")


def get_feed_forward(ffn_type: str, 
                     d_model: int, 
                     dim_ff: int, 
                     dropout: float = 0.1, 
                     dropconnect: float = 0.2) -> nn.Module:
    ffn_type = ffn_type.lower()

    if ffn_type == "relu":
        return FFN_ReLU(d_model=d_model, dim_ff=dim_ff)
    
    elif ffn_type == "gelu":
        return FFN_GELU(d_model=d_model, dim_ff=dim_ff)
    
    elif ffn_type == "geglu":
        return FFN_GEGLU(d_model=d_model, dim_ff=dim_ff)
    
    elif ffn_type == "swiglu":
        return FFN_SwiGLU(d_model=d_model, dim_ff=dim_ff)
    
    elif ffn_type == "glu":
        return FFN_GLU(d_model=d_model, dim_ff=dim_ff)
    
    elif ffn_type == "conformer":
        return FFN_Conformer(d_model=d_model, dim_ff=dim_ff, dropout=dropout)
    
    elif ffn_type == "dropconnect":
        return FFN_DropConnect(d_model=d_model, dim_ff=dim_ff, dropconnect=dropconnect)
    
    else:
        raise ValueError(f"FFN type '{ffn_type}' not supported")


def get_scheduler(scheduler_type: str,
                  optimizer: optim.Optimizer,
                  d_model: int,
                  max_lr: float = 1e-3,
                  min_lr: float = 1e-5,
                  end_lr: float = 0.0,
                  base_lr: float = 1e-3,
                  warmup_steps: int = 4000,
                  total_steps: int = 100000,
                  power: float = 1.0) -> Scheduler:
    scheduler_type = scheduler_type.lower()

    if scheduler_type in ["noam", "noamwarmup"]:
        return NoamScheduler(optimizer=optimizer, d_model=d_model, warmup_steps=warmup_steps)

    elif scheduler_type in ["cosine", "cosineanneal", "cosineannealing"]:
        return CosineAnnealingWarmup(optimizer=optimizer, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, total_steps=total_steps)

    elif scheduler_type in ["linear", "linearwarmup"]:
        return LinearWarmupDecay(optimizer=optimizer, max_lr=max_lr, warmup_steps=warmup_steps, total_steps=total_steps)

    elif scheduler_type in ["inverse_sqrt", "invsqrt", "sqrtdecay"]:
        return InverseSqrtDecay(optimizer=optimizer, base_lr=base_lr)

    elif scheduler_type in ["polynomial", "polydecay"]:
        return PolynomialDecay(optimizer=optimizer, max_lr=max_lr, end_lr=end_lr,warmup_steps=warmup_steps, total_steps=total_steps, power=power)

    elif scheduler_type in ["constant", "constantwarmup"]:
        return ConstantWarmup(optimizer=optimizer, max_lr=max_lr, warmup_steps=warmup_steps)

    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' not supported")
    
from .transformer import Transformer