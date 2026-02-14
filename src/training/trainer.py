"""
Training Module

This module implements the training regime with warm-start strategy.

Key engineering considerations:
- Warm-start encoders independently, then jointly fine-tune
- Joint cold-start causes attention collapse
- Reduced learning rate during fine-tuning
- Gradient clipping for stability
- Proper checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import time
from pathlib import Path

from .losses import HuberLoss, CombinedLoss


@dataclass
class TrainerConfig:
    """Configuration for Trainer."""
    # Training phases
    warmup_epochs: int = 10
    finetune_epochs: int = 40
    
    # Learning rates
    warmup_lr: float = 1e-3
    finetune_lr: float = 1e-4
    
    # Optimization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Loss
    huber_delta: float = 1.0
    
    # Batch size
    batch_size: int = 16
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = 'checkpoints'
    checkpoint_freq: int = 10
    
    # Logging
    log_freq: int = 10
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class NO2Dataset(Dataset):
    """
    Dataset for NO₂ prediction.
    
    Wraps satellite data and auxiliary features into PyTorch Dataset.
    """
    
    def __init__(self,
                 satellite_data: np.ndarray,
                 auxiliary_data: List[Dict],
                 targets: np.ndarray,
                 feature_engineer=None):
        """
        Initialize dataset.
        
        Args:
            satellite_data: Shape (N, T, H, W) or (N, H, W)
            auxiliary_data: List of N auxiliary feature dictionaries
            targets: Shape (N, H, W) - target NO₂ values
            feature_engineer: Optional FeatureEngineer for auxiliary data
        """
        self.satellite_data = satellite_data
        self.auxiliary_data = auxiliary_data
        self.targets = targets
        self.feature_engineer = feature_engineer
        
    def __len__(self) -> int:
        return len(self.satellite_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get satellite data
        sat = self.satellite_data[idx]
        
        # Use last time step as target, earlier for input
        if sat.ndim == 3:  # (T, H, W)
            sat_input = sat[:-1]  # (T-1, H, W) as input sequence
            target = sat[-1]  # (H, W) as target
            # For simplicity, we'll use mean over time as single input
            sat_input = sat_input.mean(axis=0, keepdims=True)  # (1, H, W)
        else:
            sat_input = sat[np.newaxis, ...]  # (1, H, W)
            target = sat  # Use same as target for demo
            
        # Get auxiliary data
        aux = self.auxiliary_data[idx]
        if self.feature_engineer is not None:
            aux_features = self.feature_engineer.transform(aux)
        else:
            # Stack available features
            features = []
            for key in ['wind_u', 'wind_v', 'blh', 'temperature', 'humidity', 'pressure']:
                if key in aux:
                    features.append(aux[key])
            aux_features = np.stack(features, axis=-1)
            
        target = self.targets[idx]
            
        return (
            torch.from_numpy(sat_input).float(),
            torch.from_numpy(aux_features).float(),
            torch.from_numpy(target).float()
        )


class Trainer:
    """
    Training manager with warm-start regime.
    
    Training strategy:
    1. Phase 1 (Warmup): Train encoders with higher learning rate
    2. Phase 2 (Fine-tune): Joint training with reduced learning rate
    
    This prevents attention collapse that occurs when training from scratch.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: TrainerConfig):
        """
        Initialize trainer.
        
        Args:
            model: CrossModalNO2Predictor or similar
            config: TrainerConfig with training parameters
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Loss function
        self.criterion = CombinedLoss(huber_delta=config.huber_delta)
        
        # Create checkpoint directory
        if config.save_checkpoints:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }
        
    def create_optimizer(self, lr: float, phase: str = 'warmup') -> optim.Optimizer:
        """
        Create optimizer for training phase.
        
        Args:
            lr: Learning rate
            phase: 'warmup' or 'finetune'
            
        Returns:
            Configured optimizer
        """
        # Different parameter groups can have different learning rates
        if phase == 'warmup':
            # Higher LR for encoders during warmup
            param_groups = [
                {'params': self.model.parameters(), 'lr': lr}
            ]
        else:
            # Lower LR for fine-tuning, especially for attention
            param_groups = [
                {'params': self.model.satellite_encoder.parameters(), 'lr': lr * 0.5},
                {'params': self.model.auxiliary_encoder.parameters(), 'lr': lr * 0.5},
                {'params': self.model.cross_attention.parameters(), 'lr': lr},
                {'params': self.model.prediction_head.parameters(), 'lr': lr},
            ]
            
        return optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )
        
    def train_epoch(self,
                    train_loader: DataLoader,
                    optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Dictionary of average loss components
        """
        self.model.train()
        epoch_losses = {'total': 0, 'huber': 0}
        num_batches = 0
        
        for batch_idx, (satellite, auxiliary, targets) in enumerate(train_loader):
            satellite = satellite.to(self.device)
            auxiliary = auxiliary.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, attention = self.model(satellite, auxiliary)
            
            # Compute loss
            losses = self.criterion(predictions, targets, attention)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
                
            optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1
            
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_losses = {'total': 0, 'huber': 0}
        num_batches = 0
        
        with torch.no_grad():
            for satellite, auxiliary, targets in val_loader:
                satellite = satellite.to(self.device)
                auxiliary = auxiliary.to(self.device)
                targets = targets.to(self.device)
                
                predictions, attention = self.model(satellite, auxiliary)
                losses = self.criterion(predictions, targets, attention)
                
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
                num_batches += 1
                
        for key in val_losses:
            val_losses[key] /= num_batches
            
        return val_losses
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              verbose: bool = True) -> Dict:
        """
        Full training with warm-start regime.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        total_epochs = self.config.warmup_epochs + self.config.finetune_epochs
        
        # Phase 1: Warmup
        if verbose:
            print(f"Phase 1: Warmup ({self.config.warmup_epochs} epochs)")
        optimizer = self.create_optimizer(self.config.warmup_lr, phase='warmup')
        
        for epoch in range(self.config.warmup_epochs):
            start_time = time.time()
            
            train_losses = self.train_epoch(train_loader, optimizer)
            epoch_time = time.time() - start_time
            
            self.history['train_loss'].append(train_losses['total'])
            self.history['epoch_times'].append(epoch_time)
            
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                self.history['val_loss'].append(val_losses['total'])
            else:
                val_losses = {'total': 0}
                
            if verbose and (epoch + 1) % self.config.log_freq == 0:
                print(f"  Epoch {epoch+1}/{self.config.warmup_epochs} - "
                      f"Train Loss: {train_losses['total']:.6f} - "
                      f"Val Loss: {val_losses['total']:.6f} - "
                      f"Time: {epoch_time:.2f}s")
                
        # Phase 2: Fine-tuning
        if verbose:
            print(f"\nPhase 2: Fine-tuning ({self.config.finetune_epochs} epochs)")
        optimizer = self.create_optimizer(self.config.finetune_lr, phase='finetune')
        
        for epoch in range(self.config.finetune_epochs):
            start_time = time.time()
            
            train_losses = self.train_epoch(train_loader, optimizer)
            epoch_time = time.time() - start_time
            
            self.history['train_loss'].append(train_losses['total'])
            self.history['epoch_times'].append(epoch_time)
            
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                self.history['val_loss'].append(val_losses['total'])
            else:
                val_losses = {'total': 0}
                
            global_epoch = self.config.warmup_epochs + epoch + 1
            
            if verbose and (epoch + 1) % self.config.log_freq == 0:
                print(f"  Epoch {epoch+1}/{self.config.finetune_epochs} - "
                      f"Train Loss: {train_losses['total']:.6f} - "
                      f"Val Loss: {val_losses['total']:.6f} - "
                      f"Time: {epoch_time:.2f}s")
                
            # Checkpointing
            if self.config.save_checkpoints and (epoch + 1) % self.config.checkpoint_freq == 0:
                self.save_checkpoint(global_epoch)
                
        return self.history
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint.get('epoch', 0)
