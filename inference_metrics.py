"""
Inference metrics tracking and benchmarking utilities
For research purposes: comparing optimized vs non-optimized inference
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import time
import torch


class InferenceMetrics:
    """
    Track and plot inference metrics for benchmarking optimizations.
    """
    
    def __init__(self, save_dir: str = './inference_benchmarks'):
        """
        Args:
            save_dir: Directory to save metrics and plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.save_dir / 'inference_metrics.json'
        
        # Load existing metrics if available
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                'runs': [],
            }
    
    def log_run(
        self,
        run_name: str,
        optimized: bool,
        prompt_length: int,
        generated_length: int,
        total_time: float,
        tokens_per_second: float,
        time_per_token: float,
        memory_used_mb: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        device: str = 'cuda',
    ):
        """
        Log a single inference run.
        
        Args:
            run_name: Name/ID of the run
            optimized: Whether optimized inference was used
            prompt_length: Length of input prompt in tokens
            generated_length: Length of generated text in tokens
            total_time: Total generation time in seconds
            tokens_per_second: Tokens generated per second
            time_per_token: Time per token in milliseconds
            memory_used_mb: Memory used in MB (optional)
            gpu_utilization: GPU utilization percentage (optional)
            device: Device used ('cuda', 'cpu', 'mps')
        """
        run_data = {
            'run_name': run_name,
            'timestamp': time.time(),
            'optimized': optimized,
            'prompt_length': prompt_length,
            'generated_length': generated_length,
            'total_time': total_time,
            'tokens_per_second': tokens_per_second,
            'time_per_token': time_per_token,
            'memory_used_mb': memory_used_mb,
            'gpu_utilization': gpu_utilization,
            'device': device,
        }
        
        self.metrics['runs'].append(run_data)
        self.save()
    
    def save(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_comparison_data(self) -> Dict:
        """
        Get comparison data for optimized vs non-optimized runs.
        
        Returns:
            Dictionary with comparison statistics
        """
        def safe_mean(values):
            """Safely compute mean, returning None if no valid values."""
            valid_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if not valid_values:
                return None
            return np.mean(valid_values)
        
        runs = self.metrics['runs']
        
        optimized_runs = [r for r in runs if r['optimized']]
        non_optimized_runs = [r for r in runs if not r['optimized']]
        
        comparison = {
            'optimized': {
                'count': len(optimized_runs),
                'avg_tokens_per_sec': np.mean([r['tokens_per_second'] for r in optimized_runs]) if optimized_runs else 0,
                'avg_time_per_token': np.mean([r['time_per_token'] for r in optimized_runs]) if optimized_runs else 0,
                'avg_total_time': np.mean([r['total_time'] for r in optimized_runs]) if optimized_runs else 0,
                'avg_memory_mb': safe_mean([r['memory_used_mb'] for r in optimized_runs]),
                'avg_gpu_util': safe_mean([r['gpu_utilization'] for r in optimized_runs]),
            },
            'non_optimized': {
                'count': len(non_optimized_runs),
                'avg_tokens_per_sec': np.mean([r['tokens_per_second'] for r in non_optimized_runs]) if non_optimized_runs else 0,
                'avg_time_per_token': np.mean([r['time_per_token'] for r in non_optimized_runs]) if non_optimized_runs else 0,
                'avg_total_time': np.mean([r['total_time'] for r in non_optimized_runs]) if non_optimized_runs else 0,
                'avg_memory_mb': safe_mean([r['memory_used_mb'] for r in non_optimized_runs]),
                'avg_gpu_util': safe_mean([r['gpu_utilization'] for r in non_optimized_runs]),
            },
        }
        
        # Calculate speedup
        if comparison['non_optimized']['avg_tokens_per_sec'] > 0:
            speedup = comparison['optimized']['avg_tokens_per_sec'] / comparison['non_optimized']['avg_tokens_per_sec']
            comparison['speedup'] = speedup
        else:
            comparison['speedup'] = None
        
        # Calculate memory reduction
        if (comparison['optimized']['avg_memory_mb'] is not None and 
            comparison['non_optimized']['avg_memory_mb'] is not None and
            comparison['non_optimized']['avg_memory_mb'] > 0):
            memory_reduction = (1 - comparison['optimized']['avg_memory_mb'] / comparison['non_optimized']['avg_memory_mb']) * 100
            comparison['memory_reduction_percent'] = memory_reduction
        else:
            comparison['memory_reduction_percent'] = None
        
        return comparison
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison charts for optimized vs non-optimized inference.
        
        Args:
            save_path: Path to save plot (default: save_dir/optimization_comparison.png)
        """
        if save_path is None:
            save_path = self.save_dir / 'optimization_comparison.png'
        
        comparison = self.get_comparison_data()
        
        if comparison['optimized']['count'] == 0 or comparison['non_optimized']['count'] == 0:
            print("⚠️  Need both optimized and non-optimized runs for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Tokens per Second
        ax1 = axes[0, 0]
        categories = ['Optimized', 'Non-Optimized']
        tokens_per_sec = [
            comparison['optimized']['avg_tokens_per_sec'],
            comparison['non_optimized']['avg_tokens_per_sec']
        ]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax1.bar(categories, tokens_per_sec, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Tokens per Second', fontsize=12)
        ax1.set_title('Generation Speed: Tokens per Second', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, tokens_per_sec):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add speedup annotation
        if comparison['speedup']:
            speedup_text = f"Speedup: {comparison['speedup']:.2f}x"
            ax1.text(0.5, 0.95, speedup_text, transform=ax1.transAxes,
                    ha='center', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Plot 2: Time per Token
        ax2 = axes[0, 1]
        time_per_token = [
            comparison['optimized']['avg_time_per_token'],
            comparison['non_optimized']['avg_time_per_token']
        ]
        bars = ax2.bar(categories, time_per_token, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Time per Token (ms)', fontsize=12)
        ax2.set_title('Latency: Time per Token', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, time_per_token):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f} ms',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 3: Total Generation Time
        ax3 = axes[1, 0]
        total_time = [
            comparison['optimized']['avg_total_time'],
            comparison['non_optimized']['avg_total_time']
        ]
        bars = ax3.bar(categories, total_time, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Total Time (seconds)', fontsize=12)
        ax3.set_title('Total Generation Time', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, total_time):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f} s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 4: Memory Usage (if available)
        ax4 = axes[1, 1]
        if comparison['optimized']['avg_memory_mb'] and comparison['non_optimized']['avg_memory_mb']:
            memory_usage = [
                comparison['optimized']['avg_memory_mb'],
                comparison['non_optimized']['avg_memory_mb']
            ]
            bars = ax4.bar(categories, memory_usage, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax4.set_ylabel('Memory Usage (MB)', fontsize=12)
            ax4.set_title('Memory Usage', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, memory_usage):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f} MB',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Add memory reduction annotation
            if comparison['memory_reduction_percent']:
                reduction_text = f"Reduction: {comparison['memory_reduction_percent']:.1f}%"
                ax4.text(0.5, 0.95, reduction_text, transform=ax4.transAxes,
                        ha='center', va='top', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, 'Memory data\nnot available', 
                    ha='center', va='center', fontsize=12,
                    transform=ax4.transAxes)
            ax4.set_title('Memory Usage', fontsize=14, fontweight='bold')
        
        plt.suptitle('Inference Optimization Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {save_path}")
        plt.close()
    
    def plot_performance_over_time(self, save_path: Optional[str] = None):
        """
        Plot performance metrics over time for research purposes.
        
        Args:
            save_path: Path to save plot (default: save_dir/performance_over_time.png)
        """
        if save_path is None:
            save_path = self.save_dir / 'performance_over_time.png'
        
        runs = self.metrics['runs']
        if len(runs) < 2:
            print("⚠️  Need at least 2 runs for time series plot")
            return
        
        # Sort by timestamp
        sorted_runs = sorted(runs, key=lambda x: x['timestamp'])
        
        optimized_times = []
        optimized_tokens_per_sec = []
        non_optimized_times = []
        non_optimized_tokens_per_sec = []
        
        for run in sorted_runs:
            if run['optimized']:
                optimized_times.append(run['timestamp'])
                optimized_tokens_per_sec.append(run['tokens_per_second'])
            else:
                non_optimized_times.append(run['timestamp'])
                non_optimized_tokens_per_sec.append(run['tokens_per_second'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if optimized_times:
            ax.plot(optimized_times, optimized_tokens_per_sec, 'o-', 
                   label='Optimized', color='#2ecc71', linewidth=2, markersize=8)
        
        if non_optimized_times:
            ax.plot(non_optimized_times, non_optimized_tokens_per_sec, 's-',
                   label='Non-Optimized', color='#e74c3c', linewidth=2, markersize=8)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Tokens per Second', fontsize=12)
        ax.set_title('Performance Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Performance over time plot saved to: {save_path}")
        plt.close()
    
    def export_to_csv(self, save_path: Optional[str] = None):
        """
        Export metrics to CSV file for analysis.
        
        Args:
            save_path: Path to save CSV (default: save_dir/inference_metrics.csv)
        """
        if save_path is None:
            save_path = self.save_dir / 'inference_metrics.csv'
        
        import csv
        
        runs = self.metrics['runs']
        if not runs:
            print("⚠️  No runs to export")
            return
        
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'run_name', 'timestamp', 'optimized', 'prompt_length', 'generated_length',
                'total_time', 'tokens_per_second', 'time_per_token', 'memory_used_mb',
                'gpu_utilization', 'device'
            ])
            
            # Data rows
            for run in runs:
                writer.writerow([
                    run['run_name'],
                    run['timestamp'],
                    run['optimized'],
                    run['prompt_length'],
                    run['generated_length'],
                    run['total_time'],
                    run['tokens_per_second'],
                    run['time_per_token'],
                    run.get('memory_used_mb', ''),
                    run.get('gpu_utilization', ''),
                    run['device'],
                ])
        
        print(f"📊 Metrics exported to CSV: {save_path}")
    
    def print_summary(self):
        """Print comparison summary."""
        comparison = self.get_comparison_data()
        
        print("\n" + "=" * 70)
        print("INFERENCE OPTIMIZATION BENCHMARK SUMMARY")
        print("=" * 70)
        
        print(f"\nOptimized Runs: {comparison['optimized']['count']}")
        if comparison['optimized']['count'] > 0:
            print(f"  Average Tokens/Second: {comparison['optimized']['avg_tokens_per_sec']:.2f}")
            print(f"  Average Time/Token: {comparison['optimized']['avg_time_per_token']:.3f} ms")
            print(f"  Average Total Time: {comparison['optimized']['avg_total_time']:.3f} s")
            if comparison['optimized']['avg_memory_mb'] is not None:
                print(f"  Average Memory: {comparison['optimized']['avg_memory_mb']:.1f} MB")
        
        print(f"\nNon-Optimized Runs: {comparison['non_optimized']['count']}")
        if comparison['non_optimized']['count'] > 0:
            print(f"  Average Tokens/Second: {comparison['non_optimized']['avg_tokens_per_sec']:.2f}")
            print(f"  Average Time/Token: {comparison['non_optimized']['avg_time_per_token']:.3f} ms")
            print(f"  Average Total Time: {comparison['non_optimized']['avg_total_time']:.3f} s")
            if comparison['non_optimized']['avg_memory_mb'] is not None:
                print(f"  Average Memory: {comparison['non_optimized']['avg_memory_mb']:.1f} MB")
        
        if comparison['speedup']:
            print(f"\n🚀 SPEEDUP: {comparison['speedup']:.2f}x faster with optimizations")
        
        if comparison['memory_reduction_percent'] is not None:
            print(f"💾 MEMORY REDUCTION: {comparison['memory_reduction_percent']:.1f}%")
        
        print("=" * 70)
