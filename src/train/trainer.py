"""
SGWCN Trainer
Handles model training, validation, and checkpointing
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm

from ..utils.visualization import plot_training_curves


class SGWCNTrainer:
    """Trainer for SGWCN model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize trainer
        
        Args:
            model: SGWCN model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Create checkpoint directory
        os.makedirs(config['save_dir'], exist_ok=True)
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on config"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler based on config"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs']
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 50),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.config.get('gradient_clip', 0.0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.config.get('gradient_clip', 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in tqdm(self.val_loader, desc='Validation'):
            data, target = data.to(self.device), target.to(self.device)
            
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config['save_dir'], 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
    
    def train(self):
        """Train model"""
        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Training on {self.device}")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Print progress
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"New best validation accuracy: {val_acc:.2f}%")
            
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(is_best)
            
            # Plot training curves
            if (epoch + 1) % self.config.get('plot_freq', 5) == 0:
                plot_training_curves(
                    self.train_losses,
                    self.val_losses,
                    self.train_accs,
                    self.val_accs,
                    save_path=os.path.join(self.config['save_dir'], 'training_curves.png')
                )
        
        # Save final checkpoint
        self.save_checkpoint()
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%") 