"""
Complete Fusion Model

This module combines all components into the complete Cross-Modal NO₂ Predictor.
It orchestrates the satellite encoder, auxiliary encoder, cross-modal attention,
and prediction head.

Architecture overview:
1. SatelliteEncoder: NO₂ field → Spatial tokens
2. AuxiliaryEncoder: Meteorology → Temporal tokens
3. CrossModalAttention: Fuse satellite with auxiliary context
4. PredictionHead: Fused tokens → NO₂ predictions
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .satellite_encoder import SatelliteEncoder, LightweightSatelliteEncoder
from .auxiliary_encoder import AuxiliaryEncoder, DeepMLPEncoder, TemporalPooler
from .cross_modal_attention import CrossModalAttention, MultiHeadCrossAttention
from .prediction_head import PredictionHead, SpatialPredictionHead, UncertaintyHead


class CrossModalNO2Predictor(nn.Module):
    """
    Complete Cross-Modal Attention Framework for NO₂ Prediction.
    
    This is the main model class that combines:
    1. Satellite Encoder: Processes NO₂ spatial fields
    2. Auxiliary Encoder: Processes meteorological time series
    3. Cross-Modal Attention: Fuses satellite queries with auxiliary context
    4. Prediction Head: Outputs NO₂ predictions
    
    The cross-modal attention allows each spatial location to attend to
    different time steps of meteorological data, learning which weather
    conditions are most relevant for predicting NO₂ at that location.
    
    Example usage:
        model = CrossModalNO2Predictor(config)
        predictions, attention = model(satellite_data, auxiliary_data)
    """
    
    def __init__(self,
                 satellite_in_channels: int = 1,
                 auxiliary_feature_dim: int = 13,
                 embed_dim: int = 256,
                 num_attention_heads: int = 8,
                 num_attention_layers: int = 2,
                 satellite_hidden_dims: list = [32, 64, 128],
                 auxiliary_num_layers: int = 4,
                 prediction_hidden_dim: int = 128,
                 dropout: float = 0.1,
                 use_lightweight: bool = False):
        """
        Initialize the Cross-Modal NO₂ Predictor.
        
        Args:
            satellite_in_channels: Channels in satellite input (1 for NO₂)
            auxiliary_feature_dim: Dimension of auxiliary feature vector
            embed_dim: Shared embedding dimension for all modules
            num_attention_heads: Number of attention heads in cross-attention
            num_attention_layers: Number of cross-attention layers
            satellite_hidden_dims: Hidden dims for satellite encoder CNN
            auxiliary_num_layers: Number of transformer layers for auxiliary
            prediction_hidden_dim: Hidden dim for prediction head
            dropout: Dropout probability
            use_lightweight: Use lightweight encoders (faster, for testing)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Satellite Encoder
        if use_lightweight:
            self.satellite_encoder = LightweightSatelliteEncoder(
                in_channels=satellite_in_channels,
                embed_dim=embed_dim,
                dropout=dropout
            )
        else:
            self.satellite_encoder = SatelliteEncoder(
                in_channels=satellite_in_channels,
                embed_dim=embed_dim,
                hidden_dims=satellite_hidden_dims,
                dropout=dropout
            )
            
        # Auxiliary Encoder
        self.auxiliary_encoder = AuxiliaryEncoder(
            feature_dim=auxiliary_feature_dim,
            embed_dim=embed_dim,
            num_layers=auxiliary_num_layers,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Cross-Modal Attention
        if num_attention_layers == 1:
            self.cross_attention = CrossModalAttention(
                embed_dim=embed_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
        else:
            self.cross_attention = MultiHeadCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_attention_heads,
                num_layers=num_attention_layers,
                dropout=dropout
            )
            
        # Prediction Head
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=prediction_hidden_dim,
            dropout=dropout
        )
        
        # Store spatial shape for reshaping predictions
        self.spatial_shape = None
        
    def forward(self,
                satellite_data: torch.Tensor,
                auxiliary_data: torch.Tensor,
                return_attention: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the full model.
        
        Args:
            satellite_data: NO₂ field of shape (B, C, H, W)
            auxiliary_data: Auxiliary features of shape (B, T, F)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - predictions: NO₂ predictions of shape (B, H, W)
            - attention: Cross-modal attention weights or None
        """
        B, C, H, W = satellite_data.shape
        self.spatial_shape = (H, W)
        
        # Encode satellite data
        satellite_tokens, _ = self.satellite_encoder(satellite_data)  # (B, H*W, embed_dim)
        
        # Encode auxiliary data
        auxiliary_tokens = self.auxiliary_encoder(auxiliary_data)  # (B, T, embed_dim)
        
        # Cross-modal attention
        if isinstance(self.cross_attention, CrossModalAttention):
            fused_tokens, attention = self.cross_attention(
                satellite_tokens, 
                auxiliary_tokens,
                return_attention=return_attention
            )
        else:
            fused_tokens, attention_list = self.cross_attention(
                satellite_tokens,
                auxiliary_tokens,
                return_all_attention=return_attention
            )
            attention = attention_list[-1] if attention_list else None
            
        # Prediction
        predictions = self.prediction_head(fused_tokens)  # (B, H*W, 1)
        
        # Reshape to spatial grid
        predictions = predictions.view(B, H, W)
        
        return predictions, attention
    
    def get_attention_maps(self,
                           satellite_data: torch.Tensor,
                           auxiliary_data: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps for visualization.
        
        Args:
            satellite_data: Shape (B, C, H, W)
            auxiliary_data: Shape (B, T, F)
            
        Returns:
            Attention weights of shape (B, heads, H*W, T)
        """
        _, attention = self.forward(satellite_data, auxiliary_data, return_attention=True)
        return attention


class ConcatenationBaseline(nn.Module):
    """
    Baseline model using simple concatenation instead of cross-attention.
    
    This serves as an ablation baseline to demonstrate the value of
    cross-modal attention over naive feature concatenation.
    """
    
    def __init__(self,
                 satellite_in_channels: int = 1,
                 auxiliary_feature_dim: int = 13,
                 embed_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.satellite_encoder = LightweightSatelliteEncoder(
            in_channels=satellite_in_channels,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        self.auxiliary_encoder = DeepMLPEncoder(
            feature_dim=auxiliary_feature_dim,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Pool auxiliary to single token
        self.aux_pooler = TemporalPooler(embed_dim, num_output_tokens=1, pool_type='mean')
        
        # Fusion is simple concatenation + MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.prediction_head = PredictionHead(embed_dim=embed_dim)
        
    def forward(self,
                satellite_data: torch.Tensor,
                auxiliary_data: torch.Tensor) -> Tuple[torch.Tensor, None]:
        B, C, H, W = satellite_data.shape
        
        # Encode
        satellite_tokens, _ = self.satellite_encoder(satellite_data)
        auxiliary_tokens = self.auxiliary_encoder(auxiliary_data)
        
        # Pool auxiliary to single global representation
        aux_pooled = self.aux_pooler(auxiliary_tokens)  # (B, 1, embed_dim)
        
        # Broadcast and concatenate
        aux_broadcast = aux_pooled.expand(-1, satellite_tokens.shape[1], -1)
        concat = torch.cat([satellite_tokens, aux_broadcast], dim=-1)
        
        # Fusion
        fused = self.fusion_mlp(concat)
        
        # Predict
        predictions = self.prediction_head(fused).view(B, H, W)
        
        return predictions, None


class SatelliteOnlyBaseline(nn.Module):
    """
    Baseline model using only satellite data (no auxiliary fusion).
    
    This ablation helps quantify the value added by auxiliary data.
    """
    
    def __init__(self,
                 satellite_in_channels: int = 1,
                 embed_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.satellite_encoder = SatelliteEncoder(
            in_channels=satellite_in_channels,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        self.prediction_head = PredictionHead(embed_dim=embed_dim)
        
    def forward(self, satellite_data: torch.Tensor, *args) -> Tuple[torch.Tensor, None]:
        B, C, H, W = satellite_data.shape
        
        satellite_tokens, _ = self.satellite_encoder(satellite_data)
        predictions = self.prediction_head(satellite_tokens).view(B, H, W)
        
        return predictions, None


def create_model_from_config(config: Dict) -> nn.Module:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Instantiated model
    """
    model_config = config.get('model', {})
    features_config = config.get('features', {})
    
    return CrossModalNO2Predictor(
        satellite_in_channels=features_config.get('satellite', {}).get('in_channels', 1),
        auxiliary_feature_dim=features_config.get('auxiliary', {}).get('feature_dim', 13),
        embed_dim=features_config.get('satellite', {}).get('embed_dim', 256),
        num_attention_heads=model_config.get('cross_attention', {}).get('num_heads', 8),
        num_attention_layers=2,
        prediction_hidden_dim=model_config.get('prediction_head', {}).get('hidden_dim', 128),
        dropout=model_config.get('satellite_encoder', {}).get('dropout', 0.1)
    )
