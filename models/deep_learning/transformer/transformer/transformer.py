import torch
import torch.nn as nn

from ..transformer import *


class EncoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_ff: int, 
                 dropout: float, 
                 norm_type: str = "layernorm", 
                 attn_type: str = "scaled_dot",
                 ffn_type: str = "relu",  
                 pre_norm: bool = False,
                 seq_len: int = 128) -> None:
        """
        Encoder Layer consisting of Multi-Head Self-Attention and Feed-Forward Network
        
        Parameters:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dim_ff: Dimension of the feed-forward network
            dropout: Dropout rate
            norm_type: Type of normalization ("layernorm", "batchnorm",...)
            attn_type: Type of attention mechanism ("scaled_dot", "linformer",...)
            ffn_type: Type of feed-forward network ("relu", "gelu",...)
            pre_norm: If True, applies layer normalization before attention and feed-forward
            seq_len: Maximum sequence length (used for certain attention types)
        """
        super().__init__()

        self.self_attn = get_attention(attn_type=attn_type, d_model=d_model, nhead=nhead, seq_len=seq_len)
        self.ffn = get_feed_forward(ffn_type=ffn_type, d_model=d_model, dim_ff=dim_ff)
        self.norm1 = get_norm(norm_type=norm_type, d_model=d_model)
        self.norm2 = get_norm(norm_type=norm_type, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, 
                src: torch.Tensor, 
                mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the Encoder Layer
        
        Parameters:
            src: Input tensor
            mask: Optional attention mask
            key_padding_mask: Optional key padding mask
        
        Returns:
            Output tensor after applying self-attention and feed-forward network
        """
        if self.pre_norm:
            src_norm = self.norm1(src)
            attn_output = self.self_attn(src_norm, src_norm, src_norm, mask, key_padding_mask)
            src = src + self.dropout(attn_output)

            src_norm = self.norm2(src)
            src = src + self.dropout(self.ffn(src_norm))
        else:
            attn_output = self.self_attn(src, src, src, mask, key_padding_mask)
            src = self.norm1(src + self.dropout(attn_output))
            src = self.norm2(src + self.dropout(self.ffn(src)))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_ff: int, 
                 dropout: float, 
                 norm_type: str = "layernorm", 
                 attn_type: str = "scaled_dot",
                 ffn_type: str = "relu", 
                 pre_norm: bool = False,
                 seq_len: int = 128) -> None:
        """
        Decoder Layer consisting of Multi-Head Self-Attention, Cross-Attention, and Feed-Forward Network
        
        Parameters:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dim_ff: Dimension of the feed-forward network
            dropout: Dropout rate
            norm_type: Type of normalization ("layernorm", "batchnorm",...)
            attn_type: Type of attention mechanism ("scaled_dot", "linformer",...)
            ffn_type: Type of feed-forward network ("relu", "gelu",...)
            pre_norm: If True, applies layer normalization before attention and feed-forward
            seq_len: Maximum sequence length (used for certain attention types)
        """
        super().__init__()

        self.self_attn = get_attention(attn_type=attn_type, d_model=d_model, nhead=nhead, seq_len=seq_len)
        self.cross_attn = get_attention(attn_type=attn_type, d_model=d_model, nhead=nhead, seq_len=seq_len)
        self.ffn = get_feed_forward(ffn_type=ffn_type, d_model=d_model, dim_ff=dim_ff)
        self.norm1 = get_norm(norm_type=norm_type, d_model=d_model)
        self.norm2 = get_norm(norm_type=norm_type, d_model=d_model)
        self.norm3 = get_norm(norm_type=norm_type, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor, 
                tgt_mask: torch.Tensor = None, 
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None, 
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the Decoder Layer
        
        Parameters:
            tgt: Target input tensor
            memory: Encoder output tensor
            tgt_mask: Optional target attention mask
            memory_mask: Optional memory attention mask
            tgt_key_padding_mask: Optional target key padding mask
            memory_key_padding_mask: Optional memory key padding mask
        
        Returns:
            Output tensor after applying self-attention, cross-attention, and feed-forward network
        """
        if self.pre_norm:
            tgt_norm = self.norm1(tgt)
            attn_output1 = self.self_attn(tgt_norm, tgt_norm, tgt_norm, tgt_mask, tgt_key_padding_mask)
            tgt = tgt + self.dropout(attn_output1)

            tgt_norm = self.norm2(tgt)
            attn_output2 = self.cross_attn(tgt_norm, memory, memory, memory_mask, memory_key_padding_mask)
            tgt = tgt + self.dropout(attn_output2)

            tgt_norm = self.norm3(tgt)
            tgt = tgt + self.dropout(self.ffn(tgt_norm))
        else:
            attn_output1 = self.self_attn(tgt, tgt, tgt, tgt_mask, tgt_key_padding_mask)
            tgt = self.norm1(tgt + self.dropout(attn_output1))

            attn_output2 = self.cross_attn(tgt, memory, memory, memory_mask, memory_key_padding_mask)
            tgt = self.norm2(tgt + self.dropout(attn_output2))

            tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 dim_ff: int = 2048,
                 num_enc_layers: int = 6,
                 num_dec_layers: int = 6,
                 max_len: int = 128,
                 dropout: float = 0.1,
                 pad_idx: int = 0,
                 pe_type: str = "sinusoidal",
                 norm_type: str = "layernorm",
                 attn_type: str = "scaled_dot",
                 ffn_type: str = "relu",
                 pre_norm: bool = False,
                 share_embedding: bool = False):
        """
        Transformer Model consisting of Encoder and Decoder stacks

        Parameters:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Dimension of the model
            nhead: Number of attention heads
            dim_ff: Dimension of the feed-forward network
            num_enc_layers: Number of encoder layers
            num_dec_layers: Number of decoder layers
            max_len: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding index for embeddings
            pe_type: Type of pe_type encoding ("sinusoidal", "learned", "none")
            norm_type: Type of normalization ("layernorm", "batchnorm",...)
            attn_type: Type of attention mechanism ("scaled_dot", "linformer",...)
            ffn_type: Type of feed-forward network ("relu", "gelu",...)
            pre_norm: If True, applies layer normalization before attention and feed-forward
            share_embedding: If True, shares weights between source and target embeddings
        """
        super().__init__()

        # Embedding and Positional
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        if share_embedding:
            self.tgt_embed.weight = self.src_embed.weight

        self.pos_encoder = get_positional_encoding(pe_type=pe_type, d_model=d_model, 
                                                   max_len=max_len, num_heads=nhead)
        self.pos_decoder = get_positional_encoding(pe_type=pe_type, d_model=d_model, 
                                                   max_len=max_len, num_heads=nhead)

        # Encoder and Decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_ff, dropout, norm_type, attn_type, ffn_type, pre_norm)
            for _ in range(num_enc_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_ff, dropout, norm_type, attn_type, ffn_type, pre_norm)
            for _ in range(num_dec_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead

    def _merge_attn_bias(self, 
                         raw_bias: torch.Tensor, 
                         q: torch.Tensor, 
                         k: torch.Tensor) -> torch.Tensor:
        """
        Merge raw attention bias into a 3D tensor acceptable by MultiheadAttention
        
        Parameters:
            raw_bias: Raw attention bias tensor (could be 4D or 3D)
            q: Query tensor 
            k: Key tensor
        
        Returns:
            Merged attention bias tensor or None
        """
        if raw_bias is None:
            return None

        bsz, q_len, _ = q.size()
        k_len = k.size(1)

        if raw_bias.dim() == 4:
            if raw_bias.size(0) == bsz:
                bias = raw_bias[:, :, :q_len, :k_len]  
                return bias.reshape(bsz * bias.size(1), q_len, k_len)
            else:
                bias = raw_bias[0]

        if raw_bias.dim() == 3:
            bias = raw_bias[:, :q_len, :k_len] 
            bias = bias.unsqueeze(0).expand(bsz, -1, -1, -1) 
            return bias.reshape(bsz * bias.size(1), q_len, k_len)  
        
        raise ValueError(f"Unsupported raw_bias shape: {raw_bias.shape}")

    def _prepare_attn_mask(self,
                           raw_bias: torch.Tensor,
                           q: torch.Tensor,
                           k: torch.Tensor,
                           causal_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Prepare the final attention mask by combining raw attention bias and causal mask

        Parameters:
            raw_bias: Raw attention bias tensor
            q: Query tensor
            k: Key tensor
            causal_mask: Causal mask tensor

        Returns:
            Final attention mask tensor or None
        """
        bsz = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)

        bias_3d = self._merge_attn_bias(raw_bias, q, k) 

        causal_3d = None
        if causal_mask is not None:
            cm = causal_mask
            cm_add = (cm.float() * -1e9) 
            causal_3d = cm_add.unsqueeze(0).expand(bsz * self.nhead, -1, -1)

            if causal_3d.size(1) != q_len or causal_3d.size(2) != k_len:
                causal_3d = causal_3d[:, :q_len, :k_len]

        if bias_3d is None and causal_3d is None:
            return None
        if bias_3d is None:
            return causal_3d
        if causal_3d is None:
            return bias_3d
        if bias_3d.size() != causal_3d.size():
            _, cq, ck = causal_3d.size()
            bias_3d = bias_3d[:, :cq, :ck]
            causal_3d = causal_3d[:, :cq, :ck]
        return bias_3d + causal_3d
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, 
                src_ids: torch.Tensor, 
                tgt_ids: torch.Tensor) -> torch.Tensor:
        # Embedding
        src_embed = self.src_embed(src_ids)
        tgt_embed = self.tgt_embed(tgt_ids)

        # Positional encoding
        if isinstance(self.pos_encoder, (SinusoidalPositionalEncoding, LearnedPositionalEncoding)):
            src = self.dropout(self.pos_encoder(src_embed))
        else:
            src = self.dropout(src_embed)  

        if isinstance(self.pos_decoder, (SinusoidalPositionalEncoding, LearnedPositionalEncoding)):
            tgt = self.dropout(self.pos_decoder(tgt_embed))
        else:
            tgt = self.dropout(tgt_embed) 

        # Encoder
        memory = src
        for layer in self.encoder_layers:
            if isinstance(self.pos_encoder, (RelativePositionalBias, ALiBiBias)):
                raw_bias = self.pos_encoder(memory)
                attn_mask = self._prepare_attn_mask(raw_bias, memory, memory, causal_mask=None)
                memory = layer(memory, mask=attn_mask, key_padding_mask=None)
            else:
                memory = layer(memory)

        # Decoder
        causal_mask = self.generate_square_subsequent_mask(tgt.size(1))
        output = tgt
        for layer in self.decoder_layers:
            if isinstance(self.pos_decoder, (RelativePositionalBias, ALiBiBias)):
                raw_self = self.pos_decoder(output)  
                self_attn_mask = self._prepare_attn_mask(raw_self, output, output, causal_mask=causal_mask)
            else:
                self_attn_mask = self._prepare_attn_mask(None, output, output, causal_mask=causal_mask)

            if isinstance(self.pos_encoder, (RelativePositionalBias, ALiBiBias)):
                raw_cross = self.pos_encoder(memory)  
                cross_attn_mask = self._prepare_attn_mask(raw_cross, output, memory, causal_mask=None)
            else:
                cross_attn_mask = None

            output = layer(output, memory,
                           tgt_mask=self_attn_mask,
                           memory_mask=cross_attn_mask,
                           tgt_key_padding_mask=None,
                           memory_key_padding_mask=None)

        logits = self.fc_out(output)
        return logits

    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer = None,
            criterion: nn.Module = None,
            scheduler_type: str = "noam",
            number_of_epochs: int = 5,
            verbose: bool = False) -> None:
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        scheduler = get_scheduler(scheduler_type=scheduler_type, optimizer=optimizer, d_model=self.d_model)

        self.train()
        for epoch in range(number_of_epochs):
            total_loss = 0.0
            for src_ids, tgt_ids in train_loader:
                optimizer.zero_grad()
                
                tgt_input = tgt_ids[:, :-1]
                tgt_target = tgt_ids[:, 1:]

                outputs = self.forward(src_ids, tgt_input) 
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_target.reshape(-1))
                
                loss.backward()
                optimizer.step()
                scheduler.step()  

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader.dataset)
            if verbose:
                print(f"Epoch [{epoch + 1}/{number_of_epochs}], Loss: {avg_loss:.4f}")
        
    def predict(self, test_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        self.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    src_ids = batch[0]
                else:
                    src_ids = batch

                src_embed = self.src_embed(src_ids)
                if isinstance(self.pos_encoder, (SinusoidalPositionalEncoding, LearnedPositionalEncoding)):
                    memory = self.dropout(self.pos_encoder(src_embed))
                else:
                    memory = self.dropout(src_embed)

                for layer in self.encoder_layers:
                    if isinstance(self.pos_encoder, (RelativePositionalBias, ALiBiBias)):
                        raw_bias = self.pos_encoder(memory)
                        attn_mask = self._prepare_attn_mask(raw_bias, memory, memory, causal_mask=None)
                        memory = layer(memory, mask=attn_mask, key_padding_mask=None)
                    else:
                        memory = layer(memory)

                batch_size = src_ids.size(0)
                tgt_ids = torch.full((batch_size, 1), 1, dtype=torch.long)

                for _ in range(self.max_len):
                    tgt_embed = self.tgt_embed(tgt_ids)
                    if isinstance(self.pos_decoder, (SinusoidalPositionalEncoding, LearnedPositionalEncoding)):
                        tgt = self.dropout(self.pos_decoder(tgt_embed))
                    else:
                        tgt = self.dropout(tgt_embed)

                    output = tgt
                    for layer in self.decoder_layers:
                        output = layer(output, memory, tgt_mask=None, memory_mask=None,
                                       tgt_key_padding_mask=None,
                                       memory_key_padding_mask=None)

                    logits = self.fc_out(output[:, -1, :])
                    next_token = logits.argmax(dim=-1, keepdim=True)
                    tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

                    if (next_token == 2).all():
                        break

                all_preds.append(tgt_ids)

        predictions = torch.cat(all_preds, dim=0)
        return predictions