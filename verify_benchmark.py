#!/usr/bin/env python3
"""
Diagnostic script to verify benchmark results and understand the optimizations.
"""
import torch
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import data module
import importlib.util
data_module_path = project_root / "data" / "__init__.py"
spec = importlib.util.spec_from_file_location("sheepop_data", data_module_path)
sheepop_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheepop_data)
SimpleTokenizer = sheepop_data.SimpleTokenizer

from models import TransformerModel
from models.optimized_attention import OptimizedMultiHeadAttention


def check_model_architecture(model):
    """Check if model uses optimized attention."""
    optimized_layers = 0
    standard_layers = 0
    
    for module in model.modules():
        if isinstance(module, OptimizedMultiHeadAttention):
            optimized_layers += 1
        elif hasattr(module, '__class__') and 'Attention' in module.__class__.__name__:
            standard_layers += 1
    
    print(f"📊 Model Architecture Check:")
    print(f"   OptimizedMultiHeadAttention layers: {optimized_layers}")
    print(f"   Standard attention layers: {standard_layers}")
    
    if optimized_layers == 0:
        print("   ⚠️  WARNING: Model does NOT use OptimizedMultiHeadAttention!")
        print("   ⚠️  KV cache optimizations may not be active.")
    else:
        print("   ✅ Model uses optimized attention layers")
    
    return optimized_layers > 0


def verify_kv_cache_usage(model, device):
    """Verify if KV cache is actually being used."""
    print("\n🔍 Verifying KV Cache Usage:")
    
    # Check if any modules have KV cache initialized
    cache_count = 0
    for module in model.modules():
        if isinstance(module, OptimizedMultiHeadAttention):
            if module.kv_cache is not None:
                cache_count += 1
    
    if cache_count == 0:
        print("   ⚠️  No KV caches found in model")
        print("   ⚠️  This suggests the optimized path may not be using KV caching")
    else:
        print(f"   ✅ Found {cache_count} KV cache(s) in model")
    
    return cache_count > 0


def run_detailed_benchmark(model, tokenizer, prompt, device, max_length=50):
    """Run detailed benchmark with more diagnostics."""
    print(f"\n🔬 Detailed Benchmark Analysis:")
    print(f"   Prompt: {prompt[:50]}...")
    print(f"   Max length: {max_length}")
    
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=device)
    
    # Non-optimized
    print("\n   🔴 Non-Optimized:")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    
    start = time.time()
    generated_std = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_std = time.time() - start
    mem_std = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else None
    
    print(f"      Time: {time_std:.4f}s")
    if mem_std:
        print(f"      Peak Memory: {mem_std:.2f} MB")
    
    # Optimized
    print("\n   🟢 Optimized:")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    
    optimizer = model.get_optimized_inference()
    start = time.time()
    generated_opt = optimizer.generate_with_cache(
        input_ids=input_ids,
        max_length=max_length,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
    )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_opt = time.time() - start
    mem_opt = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else None
    
    print(f"      Time: {time_opt:.4f}s")
    if mem_opt:
        print(f"      Peak Memory: {mem_opt:.2f} MB")
    
    # Compare
    speedup = time_std / time_opt if time_opt > 0 else 0
    print(f"\n   📈 Results:")
    print(f"      Speedup: {speedup:.2f}x")
    if mem_std and mem_opt:
        reduction = (1 - mem_opt / mem_std) * 100
        print(f"      Memory Reduction: {reduction:.1f}%")
    
    # Check if outputs are similar
    std_text = tokenizer.decode(generated_std[0].cpu().tolist())
    opt_text = tokenizer.decode(generated_opt[0].cpu().tolist())
    
    if std_text == opt_text:
        print(f"      ✅ Outputs are identical")
    else:
        print(f"      ⚠️  Outputs differ (this is normal with sampling)")
        print(f"      Standard: {std_text[:50]}...")
        print(f"      Optimized: {opt_text[:50]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify benchmark results')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--prompt', type=str, default='The future of AI', help='Test prompt')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max-length', type=int, default=50, help='Max generation length')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    tokenizer = SimpleTokenizer()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get('model_config', {})
    
    if not model_config:
        print("⚠️  No model_config in checkpoint, using defaults")
        model_config = {
            'vocab_size': tokenizer.vocab_size,
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'd_ff': 2048,
            'max_seq_len': 512,
            'dropout': 0.1,
            'activation': 'gelu',
        }
    
    model = TransformerModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("\n" + "="*70)
    print("BENCHMARK VERIFICATION")
    print("="*70)
    
    # Check architecture
    uses_optimized = check_model_architecture(model)
    
    # Verify cache usage
    verify_kv_cache_usage(model, device)
    
    # Run detailed benchmark
    run_detailed_benchmark(model, tokenizer, args.prompt, device, args.max_length)
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    
    if not uses_optimized:
        print("⚠️  The model does NOT use OptimizedMultiHeadAttention.")
        print("⚠️  The 'optimized' path may not actually use KV caching.")
        print("⚠️  Any speedup is likely from other factors (GPU warmup, etc.)")
        print("\n💡 To enable real optimizations, you need to:")
        print("   1. Set use_optimized_attention=True when creating the model")
        print("   2. Or modify the model to use optimized attention")
    else:
        print("✅ Model uses optimized attention layers")
        print("✅ KV cache optimizations should be active")
    
    print("="*70)


if __name__ == '__main__':
    main()

