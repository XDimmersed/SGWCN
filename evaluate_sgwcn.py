"""
Evaluation script for Spiking Graph Wavelet Convolution Network (SGWCN)
Comprehensive evaluation with metrics, analysis, and visualization
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import SGWCNClassifier
from src.data.dataset import create_dataloaders
from configs.sgwcn_config import SGWCNConfig


class SGWCNEvaluator:
    """Comprehensive evaluator for SGWCN model"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize evaluator
        
        Args:
            model_path: path to trained model checkpoint
            config_path: path to config file (optional)
            device: device to use for evaluation
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load configuration and model
        self.config, self.model = self._load_model(model_path, config_path)
        
        # Create test dataloader
        _, self.test_loader = create_dataloaders(
            data_root=self.config.data_root,
            batch_size=32,  # Use smaller batch size for evaluation
            num_points=self.config.num_points,
            num_workers=4,
            normalize=self.config.normalize,
            augmentation=False,  # No augmentation for evaluation
            cache_data=False
        )
        
        # Get class names
        self.class_names = self.test_loader.dataset.class_names
        self.num_classes = len(self.class_names)
        
        print(f"Loaded model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Test dataset: {len(self.test_loader.dataset)} samples")
        
    def _load_model(self, model_path: str, config_path: Optional[str]) -> Tuple[SGWCNConfig, nn.Module]:
        """Load model and configuration"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            config = SGWCNConfig.load(config_path)
        elif 'config' in checkpoint:
            config = SGWCNConfig(**checkpoint['config'])
        else:
            # Use default configuration
            print("Warning: Using default configuration")
            config = SGWCNConfig()
        
        # Create model
        model = SGWCNClassifier(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            num_classes=config.num_classes,
            num_points=config.num_points,
            num_time_steps=config.num_time_steps,
            k_neighbors=config.k_neighbors,
            chebyshev_order=config.chebyshev_order,
            beta=config.beta,
            lambda_param=config.lambda_param,
            epsilon=config.epsilon,
            tau_mem=config.tau_mem,
            theta_pos=config.theta_pos,
            theta_neg=config.theta_neg,
            dropout=config.dropout,
            use_faiss=config.use_faiss,
            readout_mode=config.readout_mode
        )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return config, model
    
    def evaluate_accuracy(self) -> Dict[str, float]:
        """Evaluate model accuracy and related metrics"""
        print("Evaluating model accuracy...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        total_time = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                start_time = time.time()
                output = self.model(data)
                inference_time = time.time() - start_time
                
                total_time += inference_time
                num_batches += 1
                
                # Collect predictions
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = np.mean(all_predictions == all_targets)
        top3_accuracy = top_k_accuracy_score(all_targets, all_probabilities, k=3)
        top5_accuracy = top_k_accuracy_score(all_targets, all_probabilities, k=5)
        
        # Per-class accuracy
        per_class_acc = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = all_targets == i
            if np.sum(class_mask) > 0:
                per_class_acc[class_name] = np.mean(all_predictions[class_mask] == all_targets[class_mask])
            else:
                per_class_acc[class_name] = 0.0
        
        # Timing statistics
        avg_inference_time = total_time / num_batches
        throughput = len(self.test_loader.dataset) / total_time
        
        metrics = {
            'accuracy': accuracy * 100,
            'top3_accuracy': top3_accuracy * 100,
            'top5_accuracy': top5_accuracy * 100,
            'per_class_accuracy': per_class_acc,
            'avg_inference_time': avg_inference_time,
            'throughput': throughput,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def comprehensive_evaluation(self, save_dir: str = 'evaluation_results'):
        """Run comprehensive evaluation and save all results"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 60)
        print("SGWCN Comprehensive Evaluation")
        print("=" * 60)
        
        # 1. Accuracy evaluation
        accuracy_metrics = self.evaluate_accuracy()
        
        print(f"\n📊 Accuracy Metrics:")
        print(f"  Overall Accuracy: {accuracy_metrics['accuracy']:.2f}%")
        print(f"  Top-3 Accuracy: {accuracy_metrics['top3_accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {accuracy_metrics['top5_accuracy']:.2f}%")
        print(f"  Inference Time: {accuracy_metrics['avg_inference_time']:.4f}s")
        print(f"  Throughput: {accuracy_metrics['throughput']:.1f} samples/s")
        
        # 2. Generate confusion matrix
        cm = confusion_matrix(accuracy_metrics['targets'], accuracy_metrics['predictions'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - SGWCN on ModelNet40')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()
        
        # 3. Classification report
        report = classification_report(
            accuracy_metrics['targets'], 
            accuracy_metrics['predictions'],
            target_names=self.class_names,
            output_dict=True
        )
        
        # 4. Save results
        results = {
            'accuracy_metrics': {k: v for k, v in accuracy_metrics.items() 
                               if k not in ['predictions', 'targets', 'probabilities']},
            'classification_report': report,
            'model_config': self.config.to_dict()
        }
        
        results_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {save_dir}")
        print("=" * 60)
        
        return results


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate SGWCN model')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', 
                       help='Directory to save results')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick evaluation with fewer samples')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SGWCNEvaluator(
        model_path=args.model_path,
        config_path=args.config,
        device=args.device
    )
    
    # Run evaluation
    if args.quick:
        print("Running quick evaluation...")
        accuracy_metrics = evaluator.evaluate_accuracy()
        print(f"Accuracy: {accuracy_metrics['accuracy']:.2f}%")
        print(f"Top-3 Accuracy: {accuracy_metrics['top3_accuracy']:.2f}%")
    else:
        evaluator.comprehensive_evaluation(save_dir=args.save_dir)


if __name__ == "__main__":
    main() 