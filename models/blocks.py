"""
Transformer building blocks: Feed-forward networks and transformer blocks
"""
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .optimized_attention import OptimizedMultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Implements two linear transformations with activation in between.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        bias: bool = False,
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout probability
            activation: Activation function ('gelu' or 'relu')
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.
    Includes residual connections and layer normalization.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        bias: bool = False,
        causal: bool = False,
        use_optimized_attention: bool = False,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
            layer_norm_eps: Epsilon for layer normalization
            bias: Whether to use bias in linear layers
            causal: Whether to use causal masking
            use_optimized_attention: Whether to use optimized attention with KV caching
        """
        super().__init__()
        
        # Self-attention with pre-norm architecture
        if use_optimized_attention:
            self.self_attn = OptimizedMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                causal=causal,
                use_flash_attention=True,
            )
        else:
            self.self_attn = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                causal=causal,
            )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            use_cache: Whether to use KV cache (for optimized attention)
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-norm self-attention with residual connection
        residual = x
        x = self.norm1(x)
        # Pass use_cache to attention layer if it's OptimizedMultiHeadAttention
        if isinstance(self.self_attn, OptimizedMultiHeadAttention):
            attn_out, _ = self.self_attn(x, x, x, mask=mask, use_cache=use_cache)
        else:
            attn_out, _ = self.self_attn(x, x, x, mask=mask)
        x = residual + self.dropout(attn_out)
        
        # Pre-norm feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        ff_out = self.feed_forward(x)
        x = residual + self.dropout(ff_out)
        
        return x


