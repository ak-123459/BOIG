# ============================================================================
# FILE 1: config.py
# ============================================================================
"""
Configuration management for wake word detection pipeline
Usage: from config import Config
"""

import os
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import yaml


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    n_dct_filters: int = 40
    n_mels: int = 40
    f_max: int = 4000
    f_min: int = 20
    n_fft: int = 480
    hop_ms: int = 10
    target_length: int = 101


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    positive_dir: str = "/content/dataset/positive"
    negative_dir: str = "/content/dataset/negative"
    cache_dir: str = "mfcc_cache"
    max_samples_per_class: Optional[int] = None
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data augmentation
    augment_train: bool = True
    noise_std: float = 0.005
    shift_range: int = 5
    scale_range: Tuple[float, float] = (0.8, 1.2)
    augmentation_prob: float = 0.3
    
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_shape: Tuple[int, int, int] = (1, 101, 40)
    num_classes: int = 2
    freeze_conv: bool = True
    dropout: float = 0.5
    pretrained_model_path: str = "pretrained_model.pt"


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Hardware
    num_workers: int = 2
    pin_memory: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True


@dataclass
class Config:
    """Master configuration"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(
            audio=AudioConfig(**config_dict.get('audio', {})),
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'audio': asdict(self.audio),
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training)
        }
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def __str__(self):
        """Pretty print configuration"""
        lines = ["="*80, "CONFIGURATION", "="*80]
        for section_name in ['audio', 'data', 'model', 'training']:
            section = getattr(self, section_name)
            lines.append(f"\n{section_name.upper()}:")
            for key, value in asdict(section).items():
                lines.append(f"  {key}: {value}")
        lines.append("="*80)
        return "\n".join(lines)
