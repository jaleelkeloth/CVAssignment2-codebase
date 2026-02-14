"""
Loss Functions

This module implements loss functions appropriate for NO₂ prediction,
with emphasis on outlier robustness.

Key engineering considerations:
- Huber or MAE loss to reduce sensitivity to extreme pollution events
- RMSE-only optimization biases toward high-pollution regions
- Optional weighting to balance urban vs rural predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HuberLoss(nn.Module):
    """
    Huber loss for robust regression.
    
    Combines the best properties of MSE and MAE:
    - Behaves like MSE for small errors (smooth gradient)
    - Behaves like MAE for large errors (robust to outliers)
    
    Loss formula:
        L = 0.5 * x^2           if |x| <= delta
        L = delta * (|x| - 0.5 * delta)   otherwise
    
    For NO₂ prediction:
    - Small errors near background levels are optimized smoothly
    - Large errors from pollution spikes don't dominate training
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold between MSE and MAE behavior
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Huber loss.
        
        Args:
            predictions: Predicted values
            targets: Target values
            mask: Optional boolean mask (True = valid)
            
        Returns:
            Loss value
        """
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        
        quadratic = 0.5 * diff ** 2
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        
        loss = torch.where(abs_diff <= self.delta, quadratic, linear)
        
        if mask is not None:
            loss = loss * mask.float()
            if self.reduction == 'mean':
                return loss.sum() / mask.float().sum()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss


class WeightedMAELoss(nn.Module):
    """
    Weighted Mean Absolute Error.
    
    Applies different weights based on pollution levels to balance
    the optimization between urban (high NO₂) and rural (low NO₂) regions.
    
    Without weighting, models tend to optimize well for high-concentration
    regions while ignoring low-concentration areas.
    """
    
    def __init__(self, 
                 low_weight: float = 1.0,
                 high_weight: float = 1.0,
                 threshold: float = 0.0):
        """
        Initialize weighted MAE loss.
        
        Args:
            low_weight: Weight for low-concentration predictions
            high_weight: Weight for high-concentration predictions
            threshold: Threshold to separate low/high (in normalized units)
        """
        super().__init__()
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.threshold = threshold
        
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted MAE.
        
        Args:
            predictions: Predicted values
            targets: Target values
            mask: Optional validity mask
            
        Returns:
            Weighted MAE loss
        """
        abs_error = torch.abs(predictions - targets)
        
        # Compute weights based on target magnitude
        weights = torch.where(
            targets > self.threshold,
            torch.full_like(targets, self.high_weight),
            torch.full_like(targets, self.low_weight)
        )
        
        weighted_error = abs_error * weights
        
        if mask is not None:
            weighted_error = weighted_error * mask.float()
            return weighted_error.sum() / (mask.float().sum() * weights[mask].mean())
        else:
            return weighted_error.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple objectives.
    
    Combines Huber loss with additional regularization terms:
    - Spatial smoothness (optional)
    - Attention entropy regularization
    """
    
    def __init__(self,
                 huber_delta: float = 1.0,
                 spatial_smooth_weight: float = 0.0,
                 attention_entropy_weight: float = 0.01):
        """
        Initialize combined loss.
        
        Args:
            huber_delta: Huber loss threshold
            spatial_smooth_weight: Weight for spatial smoothness term
            attention_entropy_weight: Weight for attention entropy regularization
        """
        super().__init__()
        self.huber = HuberLoss(delta=huber_delta)
        self.spatial_smooth_weight = spatial_smooth_weight
        self.attention_entropy_weight = attention_entropy_weight
        
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> dict:
        """
        Compute combined loss.
        
        Args:
            predictions: Shape (B, H, W)
            targets: Shape (B, H, W)
            attention_weights: Shape (B, heads, N_s, N_a)
            mask: Optional validity mask
            
        Returns:
            Dictionary with loss components and total
        """
        losses = {}
        
        # Main prediction loss
        losses['huber'] = self.huber(predictions, targets, mask)
        
        # Spatial smoothness (penalize large gradients)
        if self.spatial_smooth_weight > 0:
            grad_h = torch.abs(predictions[:, 1:, :] - predictions[:, :-1, :]).mean()
            grad_w = torch.abs(predictions[:, :, 1:] - predictions[:, :, :-1]).mean()
            losses['spatial_smooth'] = self.spatial_smooth_weight * (grad_h + grad_w)
        else:
            losses['spatial_smooth'] = torch.tensor(0.0, device=predictions.device)
            
        # Attention entropy regularization (prevent collapse)
        if attention_weights is not None and self.attention_entropy_weight > 0:
            eps = 1e-8
            entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + eps), 
                dim=-1
            ).mean()
            max_entropy = torch.log(torch.tensor(
                attention_weights.shape[-1], 
                dtype=torch.float, 
                device=attention_weights.device
            ))
            # Penalize low entropy (uniform attention is actually desired to some extent)
            losses['attention_entropy'] = -self.attention_entropy_weight * entropy / max_entropy
        else:
            losses['attention_entropy'] = torch.tensor(0.0, device=predictions.device)
            
        losses['total'] = losses['huber'] + losses['spatial_smooth'] + losses['attention_entropy']
        
        return losses


class RMSELoss(nn.Module):
    """
    Root Mean Square Error loss.
    
    Provided for comparison but NOT recommended as primary loss
    for NO₂ prediction due to sensitivity to outliers.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute RMSE loss.
        """
        mse = (predictions - targets) ** 2
        
        if mask is not None:
            mse = mse * mask.float()
            mean_mse = mse.sum() / mask.float().sum()
        else:
            mean_mse = mse.mean()
            
        return torch.sqrt(mean_mse)
