"""
Training script for Spiking Graph Wavelet Convolution Network (SGWCN)
Supports full training pipeline with validation, checkpointing, and logging
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import SGWCNClassifier
from src.data.dataset import create_dataloaders
from src.train.trainer import SGWCNTrainer
from configs.sgwcn_config import (
    SGWCNConfig, ConfigPresets, get_debug_stable_config
)


def setup_logging(config: SGWCNConfig) -> logging.Logger:
    """Setup logging configuration"""
    log_file = os.path.join(
        config.log_dir, 
        f"sgwcn_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_model(config: SGWCNConfig) -> SGWCNClassifier:
    """Create SGWCN model"""
    model = SGWCNClassifier(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        num_classes=config.num_classes,
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
    
    # Model compilation (PyTorch 2.0+)
    if config.compile_model:
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    return model


def train_sgwcn(config: SGWCNConfig, resume_path: str = None):
    """Main training function"""
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting SGWCN training with config: {config}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, test_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_points=config.num_points,
        num_workers=config.num_workers,
        normalize=config.normalize,
        augmentation=config.augmentation,
        cache_data=config.cache_data
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(config.device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SGWCNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config.to_dict(),
        device=config.device
    )
    
    # Resume from checkpoint if specified
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming training from {resume_path}")
        trainer.load_checkpoint(resume_path)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train SGWCN model")
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'fast_debug', 'high_performance', 'energy_efficient', 'debug_stable', 'rtx4090_optimized', 'rtx4090_safe'],
                       help='Configuration preset')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--config_file', type=str, help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        config = SGWCNConfig.load(args.config_file)
    else:
        # Use preset configuration
        if args.config == 'fast_debug':
            config = ConfigPresets.fast_debug()
        elif args.config == 'high_performance':
            config = ConfigPresets.high_performance()
        elif args.config == 'energy_efficient':
            config = ConfigPresets.energy_efficient()
        elif args.config == 'debug_stable':
            config = get_debug_stable_config()
        elif args.config == 'rtx4090_optimized':
            config = ConfigPresets.rtx4090_optimized()
        elif args.config == 'rtx4090_safe':
            config = ConfigPresets.rtx4090_safe()
        else:
            config = ConfigPresets.default()
    
    # Override with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.device:
        config.device = args.device
    
    print(f"Training configuration:")
    print(f"  Preset: {args.config}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")
    
    # Start training
    train_sgwcn(config, resume_path=args.resume)


if __name__ == "__main__":
    main() 