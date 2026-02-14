"""
Cross-Modal Attention Module

This module implements the core cross-modal attention mechanism where
satellite spatial tokens query auxiliary temporal/meteorological tokens.
This models the conditional dependence between modalities.

Key engineering considerations:
- This is NOT concatenation—attention weights must be inspectable
- Satellite queries attend over auxiliary keys/values
- Compute attention per spatial token
- Validate that attention doesn't collapse to uniform weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention where satellite queries attend to auxiliary keys/values.
    
    This is the core mechanism for fusing satellite NO₂ observations with
    meteorological context. Each spatial token (representing a grid cell)
    attends to all temporal auxiliary tokens to weight the importance of
    different time steps and weather conditions.
    
    Architecture:
    1. Project satellite tokens to queries
    2. Project auxiliary tokens to keys and values  
    3. Compute scaled dot-product attention
    4. Return attended features and attention weights for interpretability
    
    Attention formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Where:
        Q: Satellite spatial tokens (B, N_spatial, d)
        K, V: Auxiliary temporal tokens (B, N_temporal, d)
    
    Example usage:
        cross_attn = CrossModalAttention(embed_dim=256, num_heads=8)
        fused, weights = cross_attn(satellite_tokens, aux_tokens)
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        Initialize cross-modal attention.
        
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability on attention weights
            bias: Whether to use bias in projection layers
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection (from satellite)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Key and Value projections (from auxiliary)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for residual connection
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize projections
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.query_proj.bias is not None:
            nn.init.zeros_(self.query_proj.bias)
            nn.init.zeros_(self.key_proj.bias)
            nn.init.zeros_(self.value_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
            
    def forward(self,
                satellite_tokens: torch.Tensor,
                auxiliary_tokens: torch.Tensor,
                key_mask: Optional[torch.Tensor] = None,
                return_attention: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute cross-modal attention.
        
        Args:
            satellite_tokens: Query tokens from satellite encoder, shape (B, N_s, embed_dim)
            auxiliary_tokens: Key/Value tokens from auxiliary encoder, shape (B, N_a, embed_dim)
            key_mask: Optional mask for auxiliary tokens, shape (B, N_a), True = masked
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - attended: Fused features, shape (B, N_s, embed_dim)
            - attn_weights: Attention weights, shape (B, num_heads, N_s, N_a) or None
        """
        B, N_s, _ = satellite_tokens.shape
        _, N_a, _ = auxiliary_tokens.shape
        
        # Project to Q, K, V
        Q = self.query_proj(satellite_tokens)  # (B, N_s, embed_dim)
        K = self.key_proj(auxiliary_tokens)    # (B, N_a, embed_dim)
        V = self.value_proj(auxiliary_tokens)  # (B, N_a, embed_dim)
        
        # Reshape for multi-head attention
        # (B, N, embed_dim) -> (B, num_heads, N, head_dim)
        Q = Q.view(B, N_s, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_a, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_a, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, N_s, N_a)
        
        # Apply mask if provided
        if key_mask is not None:
            # Expand mask: (B, N_a) -> (B, 1, 1, N_a)
            key_mask = key_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_mask, float('-inf'))
            
        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attended values
        attended = torch.matmul(attn_weights, V)  # (B, heads, N_s, head_dim)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(B, N_s, self.embed_dim)
        
        # Output projection
        attended = self.out_proj(attended)
        
        # Residual connection with normalization
        attended = self.norm(satellite_tokens + attended)
        
        if return_attention:
            return attended, attn_weights
        else:
            return attended, None


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-layer cross-modal attention with feed-forward networks.
    
    Stacks multiple cross-attention layers to allow for deeper fusion.
    Each layer refines the satellite representation conditioned on auxiliary data.
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        """
        Initialize multi-layer cross-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads per layer
            num_layers: Number of cross-attention layers
            dim_feedforward: Dimension of FFN hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': CrossModalAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout
                ),
                'ffn': nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, embed_dim),
                    nn.Dropout(dropout)
                ),
                'norm': nn.LayerNorm(embed_dim)
            }))
            
    def forward(self,
                satellite_tokens: torch.Tensor,
                auxiliary_tokens: torch.Tensor,
                return_all_attention: bool = False) -> Tuple[torch.Tensor, list]:
        """
        Apply multi-layer cross-attention.
        
        Args:
            satellite_tokens: Shape (B, N_s, embed_dim)
            auxiliary_tokens: Shape (B, N_a, embed_dim)
            return_all_attention: Return attention from all layers
            
        Returns:
            Tuple of:
            - fused: Fused tokens, shape (B, N_s, embed_dim)
            - attention_weights: List of attention weights from each layer
        """
        x = satellite_tokens
        all_attn_weights = []
        
        for layer in self.layers:
            # Cross-attention
            attended, attn_weights = layer['cross_attn'](x, auxiliary_tokens)
            all_attn_weights.append(attn_weights)
            
            # FFN with residual
            x = layer['norm'](attended + layer['ffn'](attended))
            
        if return_all_attention:
            return x, all_attn_weights
        else:
            return x, [all_attn_weights[-1]]


def validate_attention_distribution(attn_weights: torch.Tensor,
                                    uniform_threshold: float = 0.1) -> dict:
    """
    Validate that attention weights are not collapsed to uniform distribution.
    
    If cross-modal attention learns uniform weights, it indicates the fusion
    is not effective—the model is not selectively attending to relevant
    auxiliary information.
    
    Args:
        attn_weights: Attention weights of shape (B, heads, N_s, N_a)
        uniform_threshold: Threshold for detecting uniform attention
        
    Returns:
        Dictionary with validation metrics:
        - is_valid: True if attention is not uniform
        - entropy: Average entropy of attention distributions
        - max_attention: Average max attention weight
        - entropy_ratio: Ratio of actual entropy to max entropy
    """
    B, H, N_s, N_a = attn_weights.shape
    
    # Compute entropy for each attention distribution
    # High entropy = uniform, Low entropy = concentrated
    eps = 1e-8
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
    max_entropy = torch.log(torch.tensor(N_a, dtype=torch.float))
    
    avg_entropy = entropy.mean().item()
    entropy_ratio = avg_entropy / max_entropy.item()
    
    # Compute average max attention weight
    max_attn = attn_weights.max(dim=-1)[0].mean().item()
    
    # Attention is considered "collapsed" if entropy is close to max
    is_valid = entropy_ratio < (1 - uniform_threshold)
    
    return {
        'is_valid': is_valid,
        'entropy': avg_entropy,
        'max_attention': max_attn,
        'entropy_ratio': entropy_ratio,
        'max_entropy': max_entropy.item()
    }


class AttentionVisualizationHelper:
    """
    Helper class for visualizing cross-modal attention patterns.
    
    Useful for interpretability analysis to understand which meteorological
    conditions the model attends to for each spatial location.
    """
    
    @staticmethod
    def reshape_attention_to_grid(attn_weights: torch.Tensor,
                                   spatial_height: int,
                                   spatial_width: int) -> torch.Tensor:
        """
        Reshape attention weights to spatial grid.
        
        Args:
            attn_weights: Shape (B, heads, H*W, N_aux)
            spatial_height: Original spatial height H
            spatial_width: Original spatial width W
            
        Returns:
            Reshaped attention of shape (B, heads, H, W, N_aux)
        """
        B, heads, N_s, N_a = attn_weights.shape
        assert N_s == spatial_height * spatial_width, \
            f"Spatial dimension mismatch: {N_s} vs {spatial_height * spatial_width}"
            
        return attn_weights.view(B, heads, spatial_height, spatial_width, N_a)
    
    @staticmethod
    def get_top_attended_features(attn_weights: torch.Tensor,
                                   feature_names: list,
                                   k: int = 3) -> list:
        """
        Get top-k attended auxiliary features for each spatial location.
        
        Args:
            attn_weights: Shape (B, heads, N_s, N_aux)
            feature_names: List of auxiliary feature/time names
            k: Number of top features to return
            
        Returns:
            List of top feature names per spatial location
        """
        # Average over heads
        avg_attn = attn_weights.mean(dim=1)  # (B, N_s, N_aux)
        
        # Get top-k indices
        _, top_indices = torch.topk(avg_attn, k=k, dim=-1)
        
        # Convert to feature names
        results = []
        for b in range(avg_attn.shape[0]):
            batch_results = []
            for s in range(avg_attn.shape[1]):
                top_features = [feature_names[i] for i in top_indices[b, s].tolist()]
                batch_results.append(top_features)
            results.append(batch_results)
            
        return results
