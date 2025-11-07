"""
Multi-Head Attention mechanism from "Attention Is All You Need"
Includes optimizations for long context and hallucination reduction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism with optional causal masking.
    
    Features:
    - Scaled dot-product attention
    - Optional causal masking for autoregressive generation
    - Efficient attention computation
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        causal: bool = False,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            causal: Whether to use causal masking
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        Q = self.q_proj(query)  # [batch_size, seq_len, d_model]
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            scores.masked_fill_(causal_mask, float('-inf'))
        
        # Apply external mask if provided
        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, d_k]
        output = output.view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Uses sinusoidal positional encoding as described in "Attention Is All You Need".
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output with positional encoding added
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - More efficient for long sequences.
    Better for long-horizon execution tasks.
    """
    
    def __init__(self, d_model: int, max_len: int = 8192):
        """
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            offset: Position offset for relative positions
            
        Returns:
            Rotated input tensor
        """
        seq_len = x.shape[1]
        device = x.device
        
        # Generate position indices
        t = torch.arange(offset, offset + seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = emb.cos()
        sin = emb.sin()
        
        # Split input into two halves
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated


