"""
Prediction Head

This module maps fused cross-modal embeddings to NO₂ predictions.
It converts per-token representations back to spatial predictions.

Key engineering considerations:
- Use regression head with Huber or MAE loss to reduce outlier sensitivity
- RMSE-only optimization biases toward high-pollution regions
- Output should match the spatial grid resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PredictionHead(nn.Module):
    """
    MLP prediction head for NO₂ regression.
    
    Takes per-token fused embeddings and predicts NO₂ concentration
    for each spatial location.
    
    Architecture:
    1. MLP layers with GELU activation
    2. Single output value per token
    
    Example usage:
        head = PredictionHead(embed_dim=256)
        predictions = head(fused_tokens)  # (B, N_s, 1)
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize prediction head.
        
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        in_dim = embed_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        # Final output layer (single value per token)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Predict NO₂ concentration.
        
        Args:
            fused_tokens: Fused embeddings of shape (B, N_s, embed_dim)
            
        Returns:
            Predictions of shape (B, N_s, 1)
        """
        return self.mlp(fused_tokens)


class SpatialPredictionHead(nn.Module):
    """
    Prediction head that outputs spatial grid predictions.
    
    Reshapes token predictions into spatial grid format and optionally
    applies spatial smoothing for coherent predictions.
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 hidden_dim: int = 128,
                 spatial_smooth: bool = False):
        super().__init__()
        
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        
        self.spatial_smooth = spatial_smooth
        if spatial_smooth:
            # Learnable smoothing kernel
            self.smooth_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            nn.init.constant_(self.smooth_conv.weight, 1/9)
            
    def forward(self, 
                fused_tokens: torch.Tensor,
                spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Predict NO₂ grid.
        
        Args:
            fused_tokens: Fused embeddings of shape (B, H*W, embed_dim)
            spatial_shape: (H, W) of output grid
            
        Returns:
            Predictions of shape (B, 1, H, W)
        """
        B = fused_tokens.shape[0]
        H, W = spatial_shape
        
        # Get per-token predictions
        predictions = self.prediction_head(fused_tokens)  # (B, H*W, 1)
        
        # Reshape to spatial grid
        predictions = predictions.view(B, H, W).unsqueeze(1)  # (B, 1, H, W)
        
        # Optional spatial smoothing
        if self.spatial_smooth:
            predictions = self.smooth_conv(predictions)
            
        return predictions


class UncertaintyHead(nn.Module):
    """
    Prediction head that estimates both mean and uncertainty.
    
    Outputs both the predicted NO₂ value and an uncertainty estimate,
    which is useful for identifying regions where the model is less confident.
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Mean prediction
        self.mean_head = nn.Linear(hidden_dim, 1)
        
        # Log variance prediction (log for numerical stability)
        self.log_var_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, fused_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and uncertainty.
        
        Args:
            fused_tokens: Fused embeddings of shape (B, N_s, embed_dim)
            
        Returns:
            Tuple of:
            - mean: Predicted NO₂, shape (B, N_s, 1)
            - std: Uncertainty estimate, shape (B, N_s, 1)
        """
        shared = self.shared_mlp(fused_tokens)
        
        mean = self.mean_head(shared)
        log_var = self.log_var_head(shared)
        
        # Convert log variance to standard deviation
        std = torch.exp(0.5 * log_var)
        
        return mean, std
