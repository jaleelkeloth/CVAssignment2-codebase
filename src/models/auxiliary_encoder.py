"""
Auxiliary Feature Encoder

This module encodes temporal and meteorological dynamics using Transformer
or deep MLP architectures. It processes time-varying auxiliary features
to create contextual embeddings for cross-modal attention.

Key engineering considerations:
- Output embedding dimensionality MUST match satellite branch
- Explicit temporal embeddings for sequence structure
- Handle variable-length temporal sequences
- Transformer encoder for capturing temporal dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    
    Adds position information to input embeddings using fixed sine/cosine
    functions at different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            Position-encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Uses learned position embeddings instead of fixed sinusoidal patterns.
    Can be more effective for shorter sequences.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer encoder block.
    
    Includes:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization (pre-norm style)
    - Residual connections
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (B, T, d_model)
        """
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        
        # Pre-norm FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


class AuxiliaryEncoder(nn.Module):
    """
    Transformer encoder for auxiliary meteorological features.
    
    Processes time-varying auxiliary data (wind, BLH, temperature, etc.)
    to create contextual embeddings. Uses Transformer architecture to
    capture temporal dependencies in weather patterns.
    
    Architecture:
    1. Input projection: feature_dim -> embed_dim
    2. Positional encoding (learnable)
    3. N Transformer encoder blocks
    4. Output: temporal tokens (B, T, embed_dim)
    
    Example usage:
        encoder = AuxiliaryEncoder(feature_dim=13, embed_dim=256)
        aux_tokens = encoder(aux_features)  # (B, T, 256)
    """
    
    def __init__(self,
                 feature_dim: int = 13,
                 embed_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 500):
        """
        Initialize the auxiliary encoder.
        
        Args:
            feature_dim: Dimension of input auxiliary features
            embed_dim: Output embedding dimension (must match satellite encoder)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of FFN hidden layer
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding (learnable works better for short sequences)
        self.pos_encoding = LearnablePositionalEncoding(
            embed_dim, max_seq_len, dropout
        )
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode auxiliary features.
        
        Args:
            x: Input tensor of shape (B, T, feature_dim)
            mask: Optional padding mask of shape (B, T)
            
        Returns:
            Encoded tokens of shape (B, T, embed_dim)
        """
        # Project input to embedding dimension
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        # Final normalization
        x = self.output_norm(x)
        
        return x


class DeepMLPEncoder(nn.Module):
    """
    Deep MLP encoder for auxiliary features.
    
    Alternative to Transformer when temporal dependencies are less important
    or for ablation studies. Processes each time step independently.
    """
    
    def __init__(self,
                 feature_dim: int = 13,
                 embed_dim: int = 256,
                 hidden_dims: list = [128, 256, 256],
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        in_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, embed_dim))
        layers.append(nn.LayerNorm(embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Temporal positional encoding
        self.pos_encoding = LearnablePositionalEncoding(embed_dim, max_len=500, dropout=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode auxiliary features.
        
        Args:
            x: Input tensor of shape (B, T, feature_dim)
            
        Returns:
            Encoded tokens of shape (B, T, embed_dim)
        """
        # Process each time step independently
        B, T, F = x.shape
        x = x.view(B * T, F)
        x = self.mlp(x)
        x = x.view(B, T, -1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        return x


class TemporalPooler(nn.Module):
    """
    Pools temporal tokens into a fixed-size representation.
    
    Useful when the downstream model expects a fixed number of auxiliary tokens
    regardless of input sequence length.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_output_tokens: int = 16,
                 pool_type: str = 'attention'):
        """
        Initialize the pooler.
        
        Args:
            embed_dim: Embedding dimension
            num_output_tokens: Number of output tokens
            pool_type: 'attention' (learned queries) or 'mean' (simple averaging)
        """
        super().__init__()
        
        self.num_output_tokens = num_output_tokens
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            # Learned query tokens
            self.query_tokens = nn.Parameter(torch.zeros(1, num_output_tokens, embed_dim))
            nn.init.trunc_normal_(self.query_tokens, std=0.02)
            
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = nn.LayerNorm(embed_dim)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool temporal tokens.
        
        Args:
            x: Input tokens of shape (B, T, embed_dim)
            
        Returns:
            Pooled tokens of shape (B, num_output_tokens, embed_dim)
        """
        B = x.shape[0]
        
        if self.pool_type == 'mean':
            # Split sequence into chunks and average
            T = x.shape[1]
            chunk_size = T // self.num_output_tokens
            if chunk_size == 0:
                return x[:, :self.num_output_tokens, :]
                
            chunks = []
            for i in range(self.num_output_tokens):
                start = i * chunk_size
                end = start + chunk_size if i < self.num_output_tokens - 1 else T
                chunks.append(x[:, start:end, :].mean(dim=1, keepdim=True))
            return torch.cat(chunks, dim=1)
            
        else:  # attention
            queries = self.query_tokens.expand(B, -1, -1)
            pooled, _ = self.attn(queries, x, x)
            return self.norm(pooled)
