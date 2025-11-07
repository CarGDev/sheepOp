"""
Configuration management
"""
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    vocab_size: int = 50257
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    activation: str = 'gelu'
    layer_norm_eps: float = 1e-5
    bias: bool = False
    tie_weights: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    max_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_amp: bool = True
    save_dir: str = './checkpoints'
    log_interval: int = 100
    eval_interval: int = 1000


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = './data'
    max_length: int = 512
    stride: Optional[int] = None
    num_workers: int = 0


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    device: str = 'cuda'
    seed: int = 42
    
    @classmethod
    def from_json(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            device=config_dict.get('device', 'cuda'),
            seed=config_dict.get('seed', 42),
        )
    
    def to_json(self, config_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'device': self.device,
            'seed': self.seed,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
    )


