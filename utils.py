"""
Utility functions for model evaluation and metrics
"""
import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np
import sys
from tqdm import tqdm


def compute_accuracy(model: nn.Module, data_loader, device: str = 'cuda') -> float:
    """
    Compute token-level accuracy.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        
    Returns:
        Accuracy score
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc="Computing accuracy",
            mininterval=0.1,
            maxinterval=1.0,
            file=sys.stderr,  # Write to stderr to avoid buffering issues
            dynamic_ncols=True,  # Auto-adjust to terminal width
            disable=False,  # Explicitly enable
        ):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids)
            predictions = torch.argmax(logits, dim=-1)
            
            # Mask out padding tokens
            mask = (labels != -100)
            correct += ((predictions == labels) * mask).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_metrics(model: nn.Module, data_loader, device: str = 'cuda') -> Dict[str, float]:
    """
    Compute various evaluation metrics.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    
    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc="Computing metrics",
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
            labels_flat = labels.view(-1)
            
            # Loss
            loss = criterion(logits, labels_flat)
            total_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = (labels_flat != -100)
            correct += ((predictions == labels_flat) * mask).sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = correct / total_tokens if total_tokens > 0 else 0.0
    perplexity = np.exp(avg_loss) if avg_loss > 0 else float('inf')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
    }


