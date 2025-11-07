"""
Example usage of optimized inference and retrieval mechanisms
Demonstrates KV caching, retrieval caching, and prefetching for RAG systems
"""
import torch
import sys
import importlib.util
from pathlib import Path

# Ensure current directory is in path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Explicitly import from local data module to avoid conflicts with stdlib 'data' module
data_module_path = project_root / "data" / "__init__.py"
spec = importlib.util.spec_from_file_location("sheepop_data", data_module_path)
sheepop_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheepop_data)
SimpleTokenizer = sheepop_data.SimpleTokenizer
create_dataloader = sheepop_data.create_dataloader

from models import TransformerModel, OptimizedInference, RetrievalCache
from models.prefetching import PrefetchDataLoader, LookaheadRetriever


def example_optimized_inference():
    """Example: Using optimized inference with KV caching."""
    print("=" * 60)
    print("Example: Optimized Inference with KV Caching")
    print("=" * 60)
    
    # Create model (example configuration)
    model = TransformerModel(
        vocab_size=128,
        d_model=512,
        num_layers=6,
        num_heads=8,
    )
    
    # Get optimized inference utility
    optimizer = model.get_optimized_inference()
    
    # Example prompt
    tokenizer = SimpleTokenizer()
    prompt = "The future of AI"
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    # Generate with KV caching (faster for autoregressive generation)
    generated = optimizer.generate_with_cache(
        input_ids=input_ids,
        max_length=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
    )
    
    print(f"Generated: {tokenizer.decode(generated[0].tolist())}")
    print()


def example_retrieval_caching():
    """Example: Using retrieval cache for similar queries."""
    print("=" * 60)
    print("Example: Retrieval Caching")
    print("=" * 60)
    
    # Create retrieval cache
    cache = RetrievalCache(max_size=1000, similarity_threshold=0.9)
    
    # Example: Simulate retrieval function
    def retrieve_documents(query: str):
        """Mock retrieval function."""
        return [
            {"doc_id": "1", "text": f"Document about {query}", "score": 0.95},
            {"doc_id": "2", "text": f"Another document about {query}", "score": 0.92},
        ]
    
    # Create query embeddings (simplified)
    query1 = "What is machine learning?"
    query1_embedding = torch.randn(128)  # Example embedding
    
    query2 = "What is deep learning?"  # Similar query
    query2_embedding = torch.randn(128)  # Example embedding (would be similar in practice)
    
    # Store first query
    import hashlib
    query1_hash = hashlib.md5(query1.encode()).hexdigest()
    results1 = retrieve_documents(query1)
    cache.set(query1_hash, query1_embedding, results1)
    
    # Retrieve from cache (should find similar query)
    query2_hash = hashlib.md5(query2.encode()).hexdigest()
    cached_results = cache.get(query2_hash, query2_embedding)
    
    if cached_results:
        print(f"Found cached results for query: {query2}")
        print(f"Retrieved {len(cached_results)} documents")
    else:
        print("Cache miss, performing retrieval...")
        results = retrieve_documents(query2)
        cache.set(query2_hash, query2_embedding, results)
    
    print()


def example_prefetching():
    """Example: Using prefetching for data loading."""
    print("=" * 60)
    print("Example: Prefetching DataLoader")
    print("=" * 60)
    
    # Create sample data
    texts = ["This is a sample text."] * 100
    tokenizer = SimpleTokenizer()
    
    # Create standard dataloader
    dataloader = create_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=32,
        max_length=512,
    )
    
    # Wrap with prefetching
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefetch_loader = PrefetchDataLoader(
        dataloader=dataloader,
        prefetch_factor=2,
        device=device,
    )
    
    print(f"Created prefetch loader with {len(prefetch_loader)} batches")
    print("Prefetching batches in background thread...")
    print()


def example_batch_generation():
    """Example: Batch generation for multiple prompts."""
    print("=" * 60)
    print("Example: Batch Generation")
    print("=" * 60)
    
    # Create model
    model = TransformerModel(
        vocab_size=128,
        d_model=512,
        num_layers=6,
        num_heads=8,
    )
    
    # Get optimized inference utility
    optimizer = model.get_optimized_inference()
    
    # Multiple prompts
    tokenizer = SimpleTokenizer()
    prompts = [
        "The future of AI",
        "Machine learning applications",
        "Deep learning advances",
    ]
    
    input_ids_list = [torch.tensor([tokenizer.encode(p)]) for p in prompts]
    
    # Generate for all prompts in batches
    results = optimizer.batch_generate(
        input_ids_list=input_ids_list,
        max_length=30,
        temperature=0.8,
        batch_size=2,
    )
    
    print(f"Generated {len(results)} responses:")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        # result is already a tensor [batch_size, seq_len], get first item if batch_size > 1
        if result.dim() > 1 and result.shape[0] > 1:
            generated_ids = result[0].tolist()
        else:
            generated_ids = result.squeeze(0).tolist() if result.dim() > 1 else result.tolist()
        generated_text = tokenizer.decode(generated_ids)
        print(f"{i+1}. Prompt: {prompt}")
        print(f"   Generated: {generated_text[:50]}...")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Optimized RAG System Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    example_optimized_inference()
    example_retrieval_caching()
    example_prefetching()
    example_batch_generation()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

