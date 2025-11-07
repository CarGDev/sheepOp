"""
Training metrics tracking and plotting utilities
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class TrainingMetrics:
    """
    Track and plot training metrics during training.
    """
    
    def __init__(self, save_dir: str = './checkpoints'):
        """
        Args:
            save_dir: Directory to save metrics and plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.save_dir / 'training_metrics.json'
        
        # Load existing metrics if available
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': [],
                'epochs': [],
                'steps': [],
            }
    
    def log(self, epoch: int, step: int, train_loss: float, 
            val_loss: Optional[float] = None, lr: Optional[float] = None):
        """
        Log training metrics.
        
        Args:
            epoch: Current epoch
            step: Current global step
            train_loss: Training loss
            val_loss: Validation loss (optional)
            lr: Learning rate (optional)
        """
        self.metrics['train_loss'].append(train_loss)
        self.metrics['epochs'].append(epoch)
        self.metrics['steps'].append(step)
        
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        else:
            self.metrics['val_loss'].append(None)
        
        if lr is not None:
            self.metrics['learning_rate'].append(lr)
        else:
            self.metrics['learning_rate'].append(None)
        
        # Save to file
        self.save()
    
    def save(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_training_curve(self, save_path: Optional[str] = None):
        """
        Plot training and validation loss curves.
        
        Args:
            save_path: Path to save plot (default: save_dir/training_curve.png)
        """
        if save_path is None:
            save_path = self.save_dir / 'training_curve.png'
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Loss curves
        ax1 = axes[0]
        steps = self.metrics['steps']
        train_loss = self.metrics['train_loss']
        val_loss = [v for v in self.metrics['val_loss'] if v is not None]
        val_steps = [steps[i] for i, v in enumerate(self.metrics['val_loss']) if v is not None]
        
        ax1.plot(steps, train_loss, label='Train Loss', color='blue', alpha=0.7)
        if val_loss:
            ax1.plot(val_steps, val_loss, label='Val Loss', color='red', alpha=0.7)
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning rate
        ax2 = axes[1]
        lr = [v for v in self.metrics['learning_rate'] if v is not None]
        lr_steps = [steps[i] for i, v in enumerate(self.metrics['learning_rate']) if v is not None]
        
        if lr:
            ax2.plot(lr_steps, lr, label='Learning Rate', color='green', alpha=0.7)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training curve saved to: {save_path}")
        plt.close()
    
    def plot_loss_by_epoch(self, save_path: Optional[str] = None):
        """
        Plot loss averaged by epoch.
        
        Args:
            save_path: Path to save plot (default: save_dir/loss_by_epoch.png)
        """
        if save_path is None:
            save_path = self.save_dir / 'loss_by_epoch.png'
        
        # Group losses by epoch
        epochs = self.metrics['epochs']
        train_loss = self.metrics['train_loss']
        
        epoch_losses = {}
        for epoch, loss in zip(epochs, train_loss):
            if epoch not in epoch_losses:
                epoch_losses[epoch] = []
            epoch_losses[epoch].append(loss)
        
        # Average losses per epoch
        epoch_nums = sorted(epoch_losses.keys())
        avg_losses = [np.mean(epoch_losses[e]) for e in epoch_nums]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_nums, avg_losses, marker='o', label='Average Train Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss by Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Loss by epoch plot saved to: {save_path}")
        plt.close()
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of training.
        
        Returns:
            Dictionary with summary statistics
        """
        train_loss = self.metrics['train_loss']
        val_loss = [v for v in self.metrics['val_loss'] if v is not None]
        
        summary = {
            'total_steps': len(train_loss),
            'total_epochs': max(self.metrics['epochs']) + 1 if self.metrics['epochs'] else 0,
            'final_train_loss': train_loss[-1] if train_loss else None,
            'best_train_loss': min(train_loss) if train_loss else None,
            'final_val_loss': val_loss[-1] if val_loss else None,
            'best_val_loss': min(val_loss) if val_loss else None,
        }
        
        return summary
    
    def print_summary(self):
        """Print training summary."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Final Train Loss: {summary['final_train_loss']:.4f}" if summary['final_train_loss'] else "Final Train Loss: N/A")
        print(f"Best Train Loss: {summary['best_train_loss']:.4f}" if summary['best_train_loss'] else "Best Train Loss: N/A")
        print(f"Final Val Loss: {summary['final_val_loss']:.4f}" if summary['final_val_loss'] else "Final Val Loss: N/A")
        print(f"Best Val Loss: {summary['best_val_loss']:.4f}" if summary['best_val_loss'] else "Best Val Loss: N/A")
        print("=" * 60)

