
# ============================================================================
# FILE 5: evaluate.py
# ============================================================================
"""
Model evaluation and testing
Usage: python evaluate.py --config config.json --checkpoint checkpoints/best_model.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging
import json

from config import Config
from data_preprocessing import DataPipeline
from model import create_wake_word_model

logger = logging.getLogger(__name__)


class Evaluator:
    """Model evaluation and metrics calculation"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions and ground truth labels"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return (
            np.array(all_predictions),
            np.array(all_probabilities),
            np.array(all_labels)
        )
    
    def calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        # Calculate metrics
        accuracy = 100. * (tp + tn) / (tp + tn + fp + fn)
        precision = 100. * tp / (tp + fp + 1e-8)
        recall = 100. * tp / (tp + fn + 1e-8)
        specificity = 100. * tn / (tn + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # False alarm rate
        far = 100. * fp / (fp + tn + 1e-8)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'false_alarm_rate': far,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_path: str = 'confusion_matrix.png'
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Confusion matrix saved: {save_path}")
    
    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: str = 'roc_curve.png'
    ):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ ROC curve saved: {save_path}")
        return roc_auc
    
    def plot_precision_recall_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: str = 'precision_recall_curve.png'
    ):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Precision-Recall curve saved: {save_path}")
        return pr_auc
    
    def generate_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = 'classification_report.txt'
    ):
        """Generate and save classification report"""
        report = classification_report(
            labels,
            predictions,
            target_names=['Negative', 'Positive'],
            digits=4
        )
        
        with open(save_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
        
        logger.info(f"  ✓ Classification report saved: {save_path}")
        print("\n" + report)
    
    def evaluate(self, output_dir: str = 'evaluation_results') -> Dict[str, float]:
        """Complete evaluation pipeline"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        # Get predictions
        predictions, probabilities, labels = self.predict()
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels)
        
        # Print metrics
        logger.info("\nTest Set Metrics:")
        logger.info(f"  Accuracy:     {metrics['accuracy']:.2f}%")
        logger.info(f"  Precision:    {metrics['precision']:.2f}%")
        logger.info(f"  Recall:       {metrics['recall']:.2f}%")
        logger.info(f"  Specificity:  {metrics['specificity']:.2f}%")
        logger.info(f"  F1 Score:     {metrics['f1_score']:.2f}%")
        logger.info(f"  FAR:          {metrics['false_alarm_rate']:.2f}%")
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"  True Positives:  {metrics['true_positives']}")
        logger.info(f"  True Negatives:  {metrics['true_negatives']}")
        logger.info(f"  False Positives: {metrics['false_positives']}")
        logger.info(f"  False Negatives: {metrics['false_negatives']}")
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        self.plot_confusion_matrix(
            labels,
            predictions,
            os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        roc_auc = self.plot_roc_curve(
            labels,
            probabilities,
            os.path.join(output_dir, 'roc_curve.png')
        )
        metrics['roc_auc'] = roc_auc
        
        pr_auc = self.plot_precision_recall_curve(
            labels,
            probabilities,
            os.path.join(output_dir, 'precision_recall_curve.png')
        )
        metrics['pr_auc'] = pr_auc
        
        # Generate classification report
        self.generate_report(
            predictions,
            labels,
            os.path.join(output_dir, 'classification_report.txt')
        )
        
        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"  ✓ Metrics saved: {metrics_path}")
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        
        return metrics


def load_checkpoint(checkpoint_path: str, config: Config, device: str = 'cpu'):
    """Load model from checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_wake_word_model(config, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("✓ Model loaded successfully")
    return model


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate wake word detection model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_json(args.config)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")
    
    # Prepare test data
    pipeline = DataPipeline(config)
    datasets = pipeline.prepare_datasets(force_recompute=False)
    dataloaders = pipeline.get_dataloaders(datasets)
    
    # Load model
    model = load_checkpoint(args.checkpoint, config, device=device)
    
    # Evaluate
    evaluator = Evaluator(model, dataloaders['test'], device=device)
    metrics = evaluator.evaluate(output_dir=args.output_dir)
    
    logger.info("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
