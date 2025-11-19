#!/usr/bin/env python3
"""
Batch benchmark script for running multiple prompts and creating trends.
Collects data across multiple prompts for research analysis.
"""
import subprocess
import argparse
import json
import sys
from pathlib import Path
import time
from typing import List


def run_benchmark(
    checkpoint: str,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    device: str = 'cuda',
    benchmark_dir: str = './inference_benchmarks',
    extra_args: List[str] = None,
):
    """Run a single benchmark."""
    cmd = [
        sys.executable, 'inference.py',
        '--checkpoint', checkpoint,
        '--prompt', prompt,
        '--max-length', str(max_length),
        '--temperature', str(temperature),
        '--device', device,
        '--benchmark',
        '--benchmark-dir', benchmark_dir,
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*70}")
    print(f"Running benchmark: {prompt[:50]}...")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error running benchmark:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def load_prompts_from_file(prompt_file: str) -> List[str]:
    """Load prompts from a text file (one prompt per line)."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple benchmarks with different prompts to create trends'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompts', type=str, nargs='+',
        help='List of prompts to benchmark'
    )
    parser.add_argument(
        '--prompt-file', type=str,
        help='File containing prompts (one per line)'
    )
    parser.add_argument(
        '--max-length', type=int, default=100,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use'
    )
    parser.add_argument(
        '--benchmark-dir', type=str, default='./inference_benchmarks',
        help='Directory to save benchmark results'
    )
    parser.add_argument(
        '--delay', type=float, default=1.0,
        help='Delay between benchmarks (seconds)'
    )
    
    args = parser.parse_args()
    
    # Collect prompts
    prompts = []
    
    if args.prompt_file:
        prompts.extend(load_prompts_from_file(args.prompt_file))
        print(f"📝 Loaded {len(prompts)} prompts from {args.prompt_file}")
    
    if args.prompts:
        prompts.extend(args.prompts)
        print(f"📝 Added {len(args.prompts)} prompts from command line")
    
    if not prompts:
        print("❌ No prompts provided! Use --prompts or --prompt-file")
        return
    
    print(f"\n✅ Total prompts to benchmark: {len(prompts)}")
    print(f"📊 Results will be saved to: {args.benchmark_dir}")
    print(f"⏱️  Delay between runs: {args.delay}s\n")
    
    # Run benchmarks
    successful = 0
    failed = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Processing prompt...")
        
        success = run_benchmark(
            checkpoint=args.checkpoint,
            prompt=prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            device=args.device,
            benchmark_dir=args.benchmark_dir,
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Delay between runs
        if i < len(prompts):
            time.sleep(args.delay)
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successful: {successful}/{len(prompts)}")
    print(f"❌ Failed: {failed}/{len(prompts)}")
    print(f"\n📊 All data saved to: {args.benchmark_dir}")
    print(f"   - JSON metrics: {args.benchmark_dir}/inference_metrics.json")
    print(f"   - CSV export: {args.benchmark_dir}/inference_metrics.csv")
    print(f"   - Comparison plots: {args.benchmark_dir}/optimization_comparison.png")
    print(f"   - Trend plot: {args.benchmark_dir}/performance_over_time.png")
    
    # Load and show summary
    metrics_file = Path(args.benchmark_dir) / 'inference_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        runs = metrics.get('runs', [])
        optimized_runs = [r for r in runs if r['optimized']]
        non_optimized_runs = [r for r in runs if not r['optimized']]
        
        if optimized_runs and non_optimized_runs:
            avg_optimized = sum(r['tokens_per_second'] for r in optimized_runs) / len(optimized_runs)
            avg_non_optimized = sum(r['tokens_per_second'] for r in non_optimized_runs) / len(non_optimized_runs)
            speedup = avg_optimized / avg_non_optimized if avg_non_optimized > 0 else 0
            
            print(f"\n📈 OVERALL PERFORMANCE:")
            print(f"   Average Optimized: {avg_optimized:.2f} tokens/sec")
            print(f"   Average Non-Optimized: {avg_non_optimized:.2f} tokens/sec")
            print(f"   Overall Speedup: {speedup:.2f}x")


if __name__ == '__main__':
    main()

