
# ============================================================================
# FILE 3: model.py
# ============================================================================
"""
Model architecture definitions
Usage: from model import WakeWordModel, load_pretrained_model
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PretrainedModel(nn.Module):
    """Original pretrained model architecture"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(20, 8))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(10, 4))
        self.output = nn.Linear(26624, 12)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class WakeWordModel(nn.Module):
    """Wake word detection model with transfer learning"""
    
    def __init__(
        self,
        pretrained_model: nn.Module,
        input_shape: Tuple[int, int, int] = (1, 101, 40),
        num_classes: int = 2,
        freeze_conv: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Copy pretrained convolutional layers
        self.conv1 = pretrained_model.conv1
        self.conv2 = pretrained_model.conv2
        
        # Freeze conv layers if specified
        if freeze_conv:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.conv2.parameters():
                param.requires_grad = False
            logger.info("  Conv layers: FROZEN")
        else:
            logger.info("  Conv layers: TRAINABLE")
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = torch.relu(self.conv1(dummy_input))
            x = torch.relu(self.conv2(x))
            flattened_size = x.view(1, -1).shape[1]
        
        logger.info(f"  Flattened size: {flattened_size}")
        
        # Add regularization and output layer
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(flattened_size, num_classes)
        
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Output layer: {flattened_size} -> {num_classes}")
    
    def forward(self, x):
        # Feature extraction (pretrained)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        # Classification (trainable)
        x = self.dropout(x)
        x = self.output(x)
        return x
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())


def load_pretrained_model(model_path: str, device: str = 'cpu') -> nn.Module:
    """Load pretrained model from checkpoint"""
    logger.info(f"Loading pretrained model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    pretrained_model = PretrainedModel()
    pretrained_model.load_state_dict(checkpoint)
    
    logger.info("✓ Pretrained model loaded successfully")
    return pretrained_model


def create_wake_word_model(config, device: str = 'cpu') -> WakeWordModel:
    """Create wake word model with transfer learning"""
    logger.info("Creating wake word model...")
    
    # Load pretrained model
    pretrained_model = load_pretrained_model(
        config.model.pretrained_model_path,
        device=device
    )
    
    # Create wake word model
    model = WakeWordModel(
        pretrained_model=pretrained_model,
        input_shape=config.model.input_shape,
        num_classes=config.model.num_classes,
        freeze_conv=config.model.freeze_conv,
        dropout=config.model.dropout
    )
    
    # Log model info
    logger.info(f"  Total parameters: {model.get_num_total_params():,}")
    logger.info(f"  Trainable parameters: {model.get_num_trainable_params():,}")
    
    return model
