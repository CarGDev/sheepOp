"""
Training utilities and training loop
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, Optional, Callable
from pathlib import Path
import json
import sys
from tqdm import tqdm
import math
from .metrics import TrainingMetrics


class Trainer:
    """
    Trainer class for language model training.
    Includes gradient accumulation, mixed precision training, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        optimizer=None,
        scheduler=None,
        device: str = 'cuda',
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        save_dir: str = './checkpoints',
        log_interval: int = 100,
        eval_interval: int = 1000,
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (if None, AdamW is used)
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            max_epochs: Maximum number of epochs
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            use_amp: Whether to use mixed precision training
            save_dir: Directory to save checkpoints
            log_interval: Logging interval
            eval_interval: Evaluation interval
        """
        # Convert device string to torch.device if needed
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Determine device type for AMP
        # Convert device string to torch.device if needed
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        device_type = self.device.type
        
        # Setup mixed precision training (only for CUDA)
        self.device_type = device_type
        self.use_amp = use_amp and device_type == 'cuda'  # Only use AMP for CUDA
        
        if self.use_amp:
            # Use new device-agnostic API
            self.scaler = torch.amp.GradScaler('cuda')
            self.autocast_dtype = torch.float16
        else:
            self.scaler = None
            self.autocast_dtype = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Training metrics tracking
        self.metrics = TrainingMetrics(save_dir=save_dir)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            mininterval=0.1,
            maxinterval=1.0,
            file=sys.stderr,  # Write to stderr to avoid buffering issues
            dynamic_ncols=True,  # Auto-adjust to terminal width
            disable=False,  # Explicitly enable
        )
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with mixed precision (only for CUDA)
                if self.use_amp:
                    with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                        logits, _ = self.model(input_ids)
                        
                        # Reshape for loss computation
                        logits = logits.view(-1, logits.size(-1))
                        labels = labels.view(-1)
                        
                        loss = self.criterion(logits, labels)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    logits, _ = self.model(input_ids)
                    
                    # Reshape for loss computation
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                    
                    loss = self.criterion(logits, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                    })
                    progress_bar.refresh()  # Force immediate refresh
                    sys.stderr.flush()  # Force flush stderr to ensure progress bar displays
                    
                    # Log metrics
                    self.metrics.log(
                        epoch=self.current_epoch,
                        step=self.global_step,
                        train_loss=avg_loss,
                        lr=lr,
                    )
        except KeyboardInterrupt:
            # Re-raise to be handled by outer try-except
            raise
        
        # Evaluation (only reached if no interruption)
        if self.val_loader is not None and self.global_step % self.eval_interval == 0:
            val_loss = self.evaluate()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Log epoch metrics
        self.metrics.log(
            epoch=self.current_epoch,
            step=self.global_step,
            train_loss=avg_loss,
            lr=self.optimizer.param_groups[0]['lr'],
        )
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(
            self.val_loader,
            desc="Evaluating",
            mininterval=0.1,
            maxinterval=1.0,
            file=sys.stderr,  # Write to stderr to avoid buffering issues
            dynamic_ncols=True,  # Auto-adjust to terminal width
            disable=False,  # Explicitly enable
        ):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                    logits, _ = self.model(input_ids)
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                    loss = self.criterion(logits, labels)
            else:
                logits, _ = self.model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Main training loop."""
        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Evaluation at end of epoch
                if self.val_loader is not None:
                    val_loss = self.evaluate()
                    print(f"Epoch {epoch + 1}: Train Loss = {train_metrics['loss']:.4f}, "
                          f"Val Loss = {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch + 1}: Train Loss = {train_metrics['loss']:.4f}")
                
                # Save checkpoint
                self.save_checkpoint()
        
            # Generate plots at end of training
            print("\n📊 Generating training plots...")
            try:
                self.metrics.plot_training_curve()
                self.metrics.plot_loss_by_epoch()
                self.metrics.print_summary()
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user!")
            print(f"💾 Saving checkpoint at epoch {self.current_epoch + 1}...")
            try:
                self.save_checkpoint()
                print(f"✅ Checkpoint saved! You can resume with:")
                print(f"   python3 train.py --data <data> --resume {self.save_dir}/checkpoint_epoch_{self.current_epoch}.pt")
            except Exception as e:
                print(f"⚠️  Warning: Could not save checkpoint: {e}")
            
            # Generate plots before exiting
            print("\n📊 Generating training plots...")
            try:
                self.metrics.plot_training_curve()
                self.metrics.plot_loss_by_epoch()
                self.metrics.print_summary()
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")
            
            # Exit cleanly instead of re-raising
            print("\n✅ Training interrupted successfully. Exiting...")
            return
    
    def save_checkpoint(self, is_best: bool = False, model_config: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # Save model config if provided
        if model_config is not None:
            checkpoint['model_config'] = model_config
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with val_loss = {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def compute_perplexity(model: nn.Module, data_loader, device: str = 'cuda') -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    num_tokens = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    
    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc="Computing perplexity",
            mininterval=0.1,
            maxinterval=1.0,
            file=sys.stderr,  # Write to stderr to avoid buffering issues
            dynamic_ncols=True,  # Auto-adjust to terminal width
            disable=False,  # Explicitly enable
        ):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_tokens += (labels != -100).sum().item()
    
    avg_loss = total_loss / num_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


