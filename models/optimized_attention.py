"""
Optimized attention mechanisms for production RAG systems
Implements KV caching, optimized attention computation, and retrieval optimizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class KVCache:
    """Key-Value cache for efficient autoregressive generation."""
    keys: torch.Tensor  # [batch_size, num_heads, seq_len, d_k]
    values: torch.Tensor  # [batch_size, num_heads, seq_len, d_k]
    
    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor):
        """Append new keys and values to the cache."""
        self.keys = torch.cat([self.keys, new_keys], dim=2)
        self.values = torch.cat([self.values, new_values], dim=2)
    
    def clear(self):
        """Clear the cache."""
        self.keys = None
        self.values = None


class OptimizedMultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention with KV caching and efficient computation.
    
    Features:
    - KV cache for autoregressive generation
    - Optimized attention computation
    - Support for incremental decoding
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        causal: bool = False,
        use_flash_attention: bool = False,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            causal: Whether to use causal masking
            use_flash_attention: Whether to use optimized flash attention (if available)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        self.use_flash_attention = use_flash_attention
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # KV cache for inference
        self.kv_cache: Optional[KVCache] = None
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model] (if None, uses query)
            value: Value tensor [batch_size, seq_len, d_model] (if None, uses query)
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            use_cache: Whether to use KV cache
            cache_position: Position in cache for incremental decoding
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        Q = self.q_proj(query)  # [batch_size, seq_len, d_model]
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Use KV cache if available and enabled
        if use_cache and self.kv_cache is not None:
            # Append new keys and values to cache
            self.kv_cache.append(K, V)
            K = self.kv_cache.keys
            V = self.kv_cache.values
            kv_seq_len = K.shape[2]
        else:
            kv_seq_len = seq_len
        
        # Compute attention scores with optimized computation
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized scaled dot product attention
            output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.causal,
            )
            attention_weights = None  # Flash attention doesn't return weights
        else:
            # Standard attention computation
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, kv_seq_len]
            
            # Apply causal mask if needed
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, kv_seq_len, device=query.device, dtype=torch.bool),
                    diagonal=1
                )
                scores.masked_fill_(causal_mask, float('-inf'))
            
            # Apply external mask if provided
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, seq_len, kv_seq_len]
                scores.masked_fill_(mask == 0, float('-inf'))
            
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
    
    def init_kv_cache(self, batch_size: int, max_length: int, device: torch.device):
        """Initialize KV cache for inference."""
        self.kv_cache = KVCache(
            keys=torch.empty(batch_size, self.num_heads, 0, self.d_k, device=device),
            values=torch.empty(batch_size, self.num_heads, 0, self.d_k, device=device),
        )
    
    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None


class RetrievalCache:
    """
    Approximate cache for retrieval results.
    Reduces expensive vector database lookups by caching similar queries.
    """
    
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.9):
        """
        Args:
            max_size: Maximum number of cached entries
            similarity_threshold: Minimum similarity to consider a cache hit
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, List[Dict]] = {}  # query_hash -> retrieved_docs
        self.query_embeddings: Dict[str, torch.Tensor] = {}  # query_hash -> embedding
        
    def get(self, query_hash: str, query_embedding: torch.Tensor) -> Optional[List[Dict]]:
        """
        Retrieve cached results if similar query exists.
        
        Args:
            query_hash: Hash of the query
            query_embedding: Embedding of the query
            
        Returns:
            Cached results if found, None otherwise
        """
        # Check exact match first
        if query_hash in self.cache:
            return self.cache[query_hash]
        
        # Check for similar queries
        best_match = None
        best_similarity = 0.0
        
        for cached_hash, cached_embedding in self.query_embeddings.items():
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                cached_embedding.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_hash
        
        if best_similarity >= self.similarity_threshold and best_match:
            return self.cache[best_match]
        
        return None
    
    def set(self, query_hash: str, query_embedding: torch.Tensor, results: List[Dict]):
        """
        Store query and results in cache.
        
        Args:
            query_hash: Hash of the query
            query_embedding: Embedding of the query
            results: Retrieved documents/results
        """
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.query_embeddings[oldest_key]
        
        self.cache[query_hash] = results
        self.query_embeddings[query_hash] = query_embedding
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.query_embeddings.clear()


class OptimizedInference:
    """
    Optimized inference utilities for production RAG systems.
    Includes prefetching, batching, and parallel processing.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: Model to use for inference
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate with KV cache for efficient autoregressive generation.
        
        Args:
            input_ids: Starting token indices [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token sequences
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize KV cache in all attention layers
        for module in self.model.modules():
            if isinstance(module, OptimizedMultiHeadAttention):
                module.init_kv_cache(batch_size, max_length, device)
        
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            logits, _ = self.model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
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
                eos_token_id = 3  # Default EOS token ID
                if next_token.item() == eos_token_id:
                    break
                
                # Early stopping: stop if padding token is generated (prevent generating padding)
                pad_token_id = 0  # Default padding token ID
                if next_token.item() == pad_token_id:
                    break
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        # Clear KV cache
        for module in self.model.modules():
            if isinstance(module, OptimizedMultiHeadAttention):
                module.clear_cache()
        
        return generated
    
    @torch.no_grad()
    def batch_generate(
        self,
        input_ids_list: List[torch.Tensor],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        batch_size: int = 8,
    ) -> List[torch.Tensor]:
        """
        Generate for multiple prompts in batches for efficiency.
        
        Args:
            input_ids_list: List of starting token sequences
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            batch_size: Batch size for processing
            
        Returns:
            List of generated sequences
        """
        results = []
        
        for i in range(0, len(input_ids_list), batch_size):
            batch = input_ids_list[i:i + batch_size]
            
            # Pad to same length
            max_len = max(seq.shape[1] for seq in batch)
            padded_batch = []
            for seq in batch:
                padding = torch.zeros(seq.shape[0], max_len - seq.shape[1], 
                                    dtype=seq.dtype, device=seq.device)
                padded_batch.append(torch.cat([seq, padding], dim=1))
            
            batch_tensor = torch.cat(padded_batch, dim=0)
            
            # Generate for batch
            generated = self.generate_with_cache(
                batch_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            results.extend([gen for gen in generated])
        
        return results

