"""
SheepOp LLM - A modern language model implementation
Optimized for production RAG systems
"""
from .transformer import TransformerModel
from .attention import MultiHeadAttention, PositionalEncoding
from .blocks import TransformerBlock, FeedForward
from .optimized_attention import (
    OptimizedMultiHeadAttention,
    RetrievalCache,
    OptimizedInference,
    KVCache,
)
from .prefetching import (
    PrefetchDataLoader,
    LookaheadRetriever,
    BatchPrefetcher,
)

__all__ = [
    'TransformerModel',
    'MultiHeadAttention',
    'PositionalEncoding',
    'TransformerBlock',
    'FeedForward',
    'OptimizedMultiHeadAttention',
    'RetrievalCache',
    'OptimizedInference',
    'KVCache',
    'PrefetchDataLoader',
    'LookaheadRetriever',
    'BatchPrefetcher',
]


