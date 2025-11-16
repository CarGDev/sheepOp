"""
Inference script for generating text
Optimized for production RAG systems with KV caching and efficient inference
"""
import torch
import argparse
from pathlib import Path
import sys
import importlib.util
import time

# Ensure current directory is in path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Explicitly import from local data module to avoid conflicts with stdlib 'data' module
data_module_path = project_root / "data" / "__init__.py"
spec = importlib.util.spec_from_file_location("sheepop_data", data_module_path)
sheepop_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheepop_data)
SimpleTokenizer = sheepop_data.SimpleTokenizer

from models import TransformerModel
from models.optimized_attention import OptimizedInference
from inference_metrics import InferenceMetrics


def load_model(checkpoint_path: str, device: str = 'cuda', tokenizer=None):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint or use defaults
    model_config = checkpoint.get('model_config', {})
    
    # If no config in checkpoint, try to infer from model state dict or use defaults
    if not model_config:
        print("Warning: No model_config found in checkpoint. Using defaults.")
        # Try to infer vocab_size from tokenizer if provided
        if tokenizer is not None:
            vocab_size = tokenizer.vocab_size
        else:
            # Default vocab size - should match your tokenizer
            vocab_size = 128  # Default for SimpleTokenizer
        
        model_config = {
            'vocab_size': vocab_size,
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'd_ff': 2048,
            'max_seq_len': 512,
            'dropout': 0.1,
            'activation': 'gelu',
        }
        print(f"Using default config with vocab_size={vocab_size}")
    
    model = TransformerModel(**model_config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def generate_text(
    model: TransformerModel,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = 'cuda',
    optimized: bool = False,
):
    """
    Generate text from a prompt.
    
    Returns:
        tuple: (generated_text, generated_ids, input_ids, generation_time)
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=device)
    
    # Measure generation time
    start_time = time.time()
    
    if optimized:
        optimizer = model.get_optimized_inference()
        generated = optimizer.generate_with_cache(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    else:
        generated = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    generated_ids = generated[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text, generated_ids, input_ids, generation_time


def get_memory_usage(device: torch.device) -> float:
    """Get current memory usage in MB."""
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    elif device.type == 'mps':
        # MPS doesn't have direct memory query, return None
        return None
    else:
        return None


def get_peak_memory_usage(device: torch.device) -> float:
    """Get peak memory usage in MB since last reset."""
    if device.type == 'cuda':
        try:
            return torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        except RuntimeError:
            return None
    elif device.type == 'mps':
        # MPS doesn't have direct memory query, return None
        return None
    else:
        return None


def benchmark_inference(
    model: TransformerModel,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    metrics: InferenceMetrics,
    run_name: str,
):
    """Run benchmark for both optimized and non-optimized inference."""
    
    def remove_trailing_padding(token_ids, pad_token_id):
        """Remove trailing padding tokens."""
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()
        return token_ids
    
    print("\n" + "=" * 70)
    print(f"BENCHMARK RUN: {run_name}")
    print("=" * 70)
    
    results = {}
    
    # Run non-optimized first
    print("\n🔴 Running NON-OPTIMIZED inference...")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    generated_text, generated_ids, input_ids, gen_time = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=str(device),
        optimized=False,
    )
    
    # Use peak memory for more accurate measurement
    memory_used = get_peak_memory_usage(device)
    
    generated_ids = remove_trailing_padding(generated_ids, tokenizer.pad_token_id)
    prompt_length = len(input_ids[0])
    generated_length = len(generated_ids) - prompt_length
    
    if generated_length > 0:
        tokens_per_sec = generated_length / gen_time
        time_per_token = (gen_time / generated_length) * 1000  # ms
    else:
        tokens_per_sec = 0
        time_per_token = 0
    
    results['non_optimized'] = {
        'text': generated_text,
        'prompt_length': prompt_length,
        'generated_length': generated_length,
        'total_time': gen_time,
        'tokens_per_sec': tokens_per_sec,
        'time_per_token': time_per_token,
        'memory_mb': memory_used,
    }
    
    print(f"  ⏱️  Total Time: {gen_time:.3f} s")
    print(f"  📊 Tokens/Second: {tokens_per_sec:.2f}")
    print(f"  ⚡ Time/Token: {time_per_token:.3f} ms")
    if memory_used:
        print(f"  💾 Memory Used: {memory_used:.1f} MB")
    print(f"  📝 Generated: {generated_text[:100]}...")
    
    # Log metrics
    metrics.log_run(
        run_name=f"{run_name}_non_optimized",
        optimized=False,
        prompt_length=prompt_length,
        generated_length=generated_length,
        total_time=gen_time,
        tokens_per_second=tokens_per_sec,
        time_per_token=time_per_token,
        memory_used_mb=memory_used,
        device=str(device),
    )
    
    # Run optimized
    print("\n🟢 Running OPTIMIZED inference...")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    generated_text, generated_ids, input_ids, gen_time = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=str(device),
        optimized=True,
    )
    
    # Use peak memory for more accurate measurement
    memory_used = get_peak_memory_usage(device)
    
    generated_ids = remove_trailing_padding(generated_ids, tokenizer.pad_token_id)
    prompt_length = len(input_ids[0])
    generated_length = len(generated_ids) - prompt_length
    
    if generated_length > 0:
        tokens_per_sec = generated_length / gen_time
        time_per_token = (gen_time / generated_length) * 1000  # ms
    else:
        tokens_per_sec = 0
        time_per_token = 0
    
    results['optimized'] = {
        'text': generated_text,
        'prompt_length': prompt_length,
        'generated_length': generated_length,
        'total_time': gen_time,
        'tokens_per_sec': tokens_per_sec,
        'time_per_token': time_per_token,
        'memory_mb': memory_used,
    }
    
    print(f"  ⏱️  Total Time: {gen_time:.3f} s")
    print(f"  📊 Tokens/Second: {tokens_per_sec:.2f}")
    print(f"  ⚡ Time/Token: {time_per_token:.3f} ms")
    if memory_used:
        print(f"  💾 Memory Used: {memory_used:.1f} MB")
    print(f"  📝 Generated: {generated_text[:100]}...")
    
    # Log metrics
    metrics.log_run(
        run_name=f"{run_name}_optimized",
        optimized=True,
        prompt_length=prompt_length,
        generated_length=generated_length,
        total_time=gen_time,
        tokens_per_second=tokens_per_sec,
        time_per_token=time_per_token,
        memory_used_mb=memory_used,
        device=str(device),
    )
    
    # Calculate speedup
    if results['non_optimized']['tokens_per_sec'] > 0:
        speedup = results['optimized']['tokens_per_sec'] / results['non_optimized']['tokens_per_sec']
        print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster with optimizations")
    
    if results['non_optimized']['memory_mb'] and results['optimized']['memory_mb']:
        memory_reduction = (1 - results['optimized']['memory_mb'] / results['non_optimized']['memory_mb']) * 100
        print(f"💾 MEMORY REDUCTION: {memory_reduction:.1f}%")
    
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate text with SheepOp LLM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text')
    parser.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p (nucleus) sampling')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--optimized', action='store_true', help='Use optimized inference with KV caching')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparing optimized vs non-optimized inference (for research)')
    parser.add_argument('--benchmark-dir', type=str, default='./inference_benchmarks', help='Directory to save benchmark results')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create tokenizer first (needed for vocab_size)
    tokenizer = SimpleTokenizer()
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device=device, tokenizer=tokenizer)
    print("Model loaded!")
    
    # Check if benchmarking mode
    if args.benchmark:
        print("\n🔬 BENCHMARK MODE: Comparing optimized vs non-optimized inference")
        print("=" * 70)
        
        # Initialize metrics
        metrics = InferenceMetrics(save_dir=args.benchmark_dir)
        
        # Run benchmark
        run_name = f"run_{int(time.time())}"
        results = benchmark_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            metrics=metrics,
            run_name=run_name,
        )
        
        # Generate plots and summary
        print("\n📊 Generating comparison plots and data...")
        metrics.plot_comparison()
        metrics.plot_performance_over_time()
        metrics.export_to_csv()
        metrics.print_summary()
        
        print(f"\n✅ Benchmark complete! Results saved to: {args.benchmark_dir}")
        print(f"   - JSON metrics: {args.benchmark_dir}/inference_metrics.json")
        print(f"   - CSV export: {args.benchmark_dir}/inference_metrics.csv")
        print(f"   - Comparison plot: {args.benchmark_dir}/optimization_comparison.png")
        print(f"   - Performance plot: {args.benchmark_dir}/performance_over_time.png")
        
        return
    
    # Normal inference mode
    use_optimized = args.optimized if hasattr(args, 'optimized') else False
    
    if use_optimized:
        print("Using optimized inference with KV caching...")
        optimizer = model.get_optimized_inference()
    else:
        optimizer = None
    
    # Encode prompt
    input_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([input_ids], device=device)
    
    # Generate text
    print(f"Prompt: {args.prompt}")
    print("Generating...")
    
    # Filter out padding tokens from the end of generated sequence
    def remove_trailing_padding(token_ids, pad_token_id):
        """Remove trailing padding tokens."""
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()
        return token_ids
    
    if optimizer is not None:
        # Use optimized generation with KV cache
        generated = optimizer.generate_with_cache(
            input_ids=input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        generated_ids = generated[0].cpu().tolist()
        # Remove trailing padding
        generated_ids = remove_trailing_padding(generated_ids, tokenizer.pad_token_id)
        print(f"Generated {len(generated_ids)} tokens (input had {len(input_ids[0])} tokens, after removing padding)")
    else:
        # Use standard generation
        generated = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
        )
        generated_ids = generated[0].cpu().tolist()
        # Remove trailing padding
        generated_ids = remove_trailing_padding(generated_ids, tokenizer.pad_token_id)
        print(f"Generated {len(generated_ids)} tokens (input had {len(input_ids[0])} tokens, after removing padding)")
    
    # Debug: Show some token statistics
    vocab_size = tokenizer.vocab_size
    valid_tokens = sum(1 for tid in generated_ids if tid in tokenizer.inv_vocab)
    unk_tokens = sum(1 for tid in generated_ids if tid not in tokenizer.inv_vocab)
    pad_tokens = sum(1 for tid in generated_ids if tid == tokenizer.pad_token_id)
    
    print(f"Token statistics:")
    print(f"  Valid tokens: {valid_tokens}/{len(generated_ids)}")
    print(f"  Unknown tokens: {unk_tokens}")
    print(f"  Pad tokens: {pad_tokens}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Token ID range: {min(generated_ids) if generated_ids else 'N/A'} - {max(generated_ids) if generated_ids else 'N/A'}")
    
    # Show first 20 token IDs for debugging
    print(f"  First 20 token IDs: {generated_ids[:20]}")
    
    generated_text = tokenizer.decode(generated_ids)
    
    print(f"\nGenerated: {generated_text}")
    print(f"Generated length: {len(generated_text)} characters")


if __name__ == '__main__':
    main()


