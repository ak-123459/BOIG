
# ============================================================================
# FILE 4: train.py
# ============================================================================
"""
Training pipeline with monitoring and checkpointing
Usage: python train.py --config config.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Tuple
import logging

from config import Config
from data_preprocessing import DataPipeline
from model import create_wake_word_model

logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training with monitoring"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            verbose=True
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # For metrics calculation
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        pbar = tqdm(self.val_loader, desc='Validating', leave=False)
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Calculate confusion matrix components
                true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
                true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Calculate metrics
        precision = 100. * true_positives / (true_positives + false_positives + 1e-8)
        recall = 100. * true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = 100. * true_negatives / (true_negatives + false_positives + 1e-8)
        
        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        filepath = os.path.join(self.config.training.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }, filepath)
        logger.info(f"  ✓ Checkpoint saved: {filepath}")
    
    def train(self, num_epochs: Optional[int] = None):
        """Complete training loop"""
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        
        for epoch in range(num_epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_metrics = self.validate()
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(new_lr)
            
            # Print metrics
            logger.info(f"\nEpoch [{epoch+1}/{num_epochs}]")
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            logger.info(f"  Precision:  {val_metrics['precision']:.2f}% | Recall: {val_metrics['recall']:.2f}%")
            logger.info(f"  F1 Score:   {val_metrics['f1']:.2f}% | Specificity: {val_metrics['specificity']:.2f}%")
            
            if old_lr != new_lr:
                logger.info(f"  📉 Learning rate: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement ({self.patience_counter}/{self.config.training.early_stopping_patience})")
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"\n⚠ Early stopping triggered!")
                break
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"  Best Val Loss: {self.best_val_loss:.4f}")
        logger.info(f"  Best Val Acc:  {self.best_val_acc:.2f}%")
        logger.info("="*80)


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train wake word detection model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--force-recompute', action='store_true', help='Force recompute features')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_json(args.config)
    print(config)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")
    
    # Prepare data
    pipeline = DataPipeline(config)
    datasets = pipeline.prepare_datasets(force_recompute=args.force_recompute)
    dataloaders = pipeline.get_dataloaders(datasets)
    
    # Create model
    model = create_wake_word_model(config, device=device)
    
    # Train model
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config,
        device=device
    )
    trainer.train()
    
    logger.info("✅ Training pipeline complete!")


if __name__ == "__main__":
    main()

