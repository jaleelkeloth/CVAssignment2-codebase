"""
Satellite Feature Encoder

This module extracts spatial structure and gradients from NO₂ fields using
convolutional neural networks. The architecture is designed to match
atmospheric transport scales (10-50 km).

Key engineering considerations:
- Receptive field should match atmospheric transport scales (~50km at 0.1° resolution)
- Avoid excessively deep architectures (causes spatial oversmoothing)
- Multi-scale feature extraction to capture both local patterns and larger plumes
- Output: spatial tokens at each grid cell for cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """
    Basic convolutional block with normalization and activation.
    
    Includes:
    - Convolution
    - BatchNorm or LayerNorm
    - Activation (GELU)
    - Optional dropout
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dropout: float = 0.1,
                 use_layer_norm: bool = False):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_layer_norm  # No bias if using norm
        )
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            # LayerNorm expects (N, C, H, W) -> normalize over (C, H, W)
            self.norm = nn.GroupNorm(1, out_channels)  # Equivalent to LayerNorm for conv
        else:
            self.norm = nn.BatchNorm2d(out_channels)
            
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual convolutional block.
    
    Uses skip connections to facilitate gradient flow and prevent
    degradation in deeper networks.
    """
    
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # Skip connection
        out = self.activation(out)
        
        return out


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale convolutional block that processes input at multiple receptive fields.
    
    Uses parallel convolutions with different kernel sizes to capture:
    - Local emissions (3x3)
    - Medium-scale transport (5x5)
    - Large-scale plumes (7x7)
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.1):
        super().__init__()
        
        self.branches = nn.ModuleList()
        n_branches = len(kernel_sizes)
        branch_channels = out_channels // n_branches
        
        for k in kernel_sizes:
            self.branches.append(
                ConvBlock(
                    in_channels,
                    branch_channels,
                    kernel_size=k,
                    padding=k // 2,
                    dropout=dropout
                )
            )
            
        # Projection to merge branches (handle any remainder in channel division)
        total_branch_channels = branch_channels * n_branches
        self.projection = nn.Conv2d(total_branch_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through all branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate along channel dimension
        merged = torch.cat(branch_outputs, dim=1)
        
        # Project to output channels
        out = self.projection(merged)
        
        return out


class SatelliteEncoder(nn.Module):
    """
    CNN encoder for satellite NO₂ fields.
    
    Architecture designed for atmospheric transport characteristics:
    - Multi-scale feature extraction (3x3, 5x5, 7x7 kernels at 0.1° resolution)
    - Moderate depth (avoids oversmoothing)
    - Progressive channel expansion
    - Output: spatial tokens (B, H*W, embed_dim) for cross-attention
    
    At 0.1° resolution:
    - 3x3 kernel ≈ 30km receptive field (local emissions)
    - 5x5 kernel ≈ 50km receptive field (urban plumes)
    - 7x7 kernel ≈ 70km receptive field (regional transport)
    
    Example usage:
        encoder = SatelliteEncoder(in_channels=1, embed_dim=256)
        satellite_tokens = encoder(no2_field)  # (B, H*W, 256)
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 embed_dim: int = 256,
                 hidden_dims: list = [32, 64, 128],
                 kernel_sizes: list = [3, 5, 7],
                 num_residual_blocks: int = 2,
                 dropout: float = 0.1):
        """
        Initialize the satellite encoder.
        
        Args:
            in_channels: Number of input channels (1 for NO₂ only)
            embed_dim: Output embedding dimension (must match auxiliary encoder)
            hidden_dims: List of hidden channel dimensions for progressive expansion
            kernel_sizes: Kernel sizes for multi-scale feature extraction
            num_residual_blocks: Number of residual blocks at each scale
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Initial multi-scale feature extraction
        self.input_block = MultiScaleConvBlock(
            in_channels, 
            hidden_dims[0], 
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        # Progressive encoder blocks with residual connections
        self.encoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            # Transition block (channel expansion, NO downsampling)
            # We preserve spatial resolution for dense prediction
            self.encoder_blocks.append(
                ConvBlock(hidden_dims[i], hidden_dims[i+1], dropout=dropout)
            )
            
            # Residual blocks at this scale
            for _ in range(num_residual_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(hidden_dims[i+1], dropout=dropout)
                )
                
        # Final projection to embedding dimension
        self.final_conv = nn.Conv2d(hidden_dims[-1], embed_dim, kernel_size=1)
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding for spatial tokens
        self.pos_encoding = nn.Parameter(torch.zeros(1, 4096, embed_dim))  # Max 64x64
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode satellite NO₂ field.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C is in_channels
            
        Returns:
            Tuple of:
            - tokens: Spatial tokens of shape (B, H*W, embed_dim)
            - feature_map: Feature map of shape (B, embed_dim, H, W)
        """
        B, C, H, W = x.shape
        
        # Multi-scale input processing
        x = self.input_block(x)
        
        # Progressive encoding
        for block in self.encoder_blocks:
            x = block(x)
            
        # Final projection
        x = self.final_conv(x)  # (B, embed_dim, H, W)
        feature_map = x
        
        # Convert to tokens: (B, embed_dim, H, W) -> (B, H*W, embed_dim)
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        
        # Apply layer norm
        tokens = self.final_norm(tokens)
        
        # Add positional encoding
        num_tokens = H * W
        pos = self.pos_encoding[:, :num_tokens, :]
        tokens = tokens + pos
        
        return tokens, feature_map
    
    def get_receptive_field(self) -> int:
        """
        Estimate the effective receptive field in grid cells.
        
        Returns:
            Approximate receptive field size
        """
        # Rough estimate based on kernel sizes and depth
        # For this architecture: ~15-20 grid cells ≈ 150-200 km at 0.1° resolution
        return 20


class LightweightSatelliteEncoder(nn.Module):
    """
    Lightweight version of satellite encoder for faster training/inference.
    
    Uses a simple 3-layer CNN without multi-scale branches. Useful for
    initial experiments and ablation studies.
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 embed_dim: int = 256,
                 hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim, kernel_size=5, padding=2, dropout=dropout),
            ConvBlock(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, dropout=dropout),
            ConvBlock(hidden_dim * 2, embed_dim, kernel_size=3, padding=1, dropout=dropout),
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 4096, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        feature_map = self.encoder(x)  # (B, embed_dim, H, W)
        
        tokens = feature_map.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        tokens = self.final_norm(tokens)
        
        num_tokens = H * W
        tokens = tokens + self.pos_encoding[:, :num_tokens, :]
        
        return tokens, feature_map
