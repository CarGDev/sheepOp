"""
Complete Transformer model for language modeling
Incorporates best practices from multiple research papers
Optimized for production RAG systems with KV caching and efficient inference
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .blocks import TransformerBlock
from .attention import PositionalEncoding
from .optimized_attention import OptimizedMultiHeadAttention, RetrievalCache, OptimizedInference


class TransformerModel(nn.Module):
    """
    Full Transformer Language Model.
    
    Features:
    - Multi-head self-attention
    - Positional encoding
    - Layer normalization
    - Residual connections
    - Causal masking for autoregressive generation
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        bias: bool = False,
        tie_weights: bool = True,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function ('gelu' or 'relu')
            layer_norm_eps: Epsilon for layer normalization
            bias: Whether to use bias in linear layers
            tie_weights: Whether to tie input and output embeddings
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
        )
        
        # Transformer blocks (use optimized attention if available)
        # Note: Set use_optimized_attention=True for production inference
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                bias=bias,
                causal=True,  # Causal masking for autoregressive generation
                use_optimized_attention=False,  # Set to True for inference optimizations
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=bias)
        
        # Optionally tie weights
        if tie_weights:
            self.output_proj.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Retrieval cache for RAG systems
        self.retrieval_cache = RetrievalCache(max_size=1000, similarity_threshold=0.9)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following best practices."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize output projection
        if self.output_proj.weight is not self.token_embedding.weight:
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer model.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            attention_weights: Optional attention weights
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, device=input_ids.device, dtype=torch.bool
            )
        
        # Expand mask for attention
        # [batch_size, seq_len] -> [batch_size, seq_len, seq_len]
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        
        # Apply attention mask (invert: 1 for valid, 0 for masked)
        attention_mask = attention_mask.float()
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_proj(x)  # [batch_size, seq_len, vocab_size]
        
        return logits, None
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            input_ids: Starting token indices [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            
        Returns:
            Generated token sequences
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or take argmax
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Early stopping: stop if EOS or padding token is generated (for batch_size=1)
                if batch_size == 1:
                    eos_token_id = getattr(self, 'eos_token_id', None) or 3  # Default EOS token
                    if next_token.item() == eos_token_id:
                        break
                    
                    # Early stopping: stop if padding token is generated (prevent generating padding)
                    if pad_token_id is not None and next_token.item() == pad_token_id:
                        break
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_optimized_inference(self) -> OptimizedInference:
        """
        Get optimized inference utility with KV caching and batching.
        
        Returns:
            OptimizedInference instance
        """
        return OptimizedInference(self, next(self.parameters()).device)
    
    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


