"""
Auxiliary Feature Engineering Module

This module encodes contextual drivers of NO₂ formation and transport.
It transforms raw meteorological and land-use data into model-ready features
with appropriate encoding for cyclical and categorical variables.

Key engineering considerations:
- Wind vectors encoded as sine/cosine to preserve direction continuity
- Boundary layer height normalized relative to typical ranges
- Land-use encoded via learnable embeddings (not one-hot)
- Static features broadcast spatially with explicit masks
- Temporal features encoded cyclically
"""

import numpy as np
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class FeatureEngineer:
    """
    Transforms raw auxiliary features into model-ready representations.
    
    This class handles:
    1. Wind vector encoding (sine/cosine direction + magnitude)
    2. Boundary layer height normalization
    3. Temporal feature encoding (diurnal + seasonal cycles)
    4. Pressure and temperature normalization
    
    Example usage:
        engineer = FeatureEngineer()
        features = engineer.transform(aux_data)
    """
    
    def __init__(self,
                 blh_range: Tuple[float, float] = (100, 3000),
                 temp_range: Tuple[float, float] = (250, 320),
                 pressure_range: Tuple[float, float] = (950, 1050)):
        """
        Initialize the feature engineer.
        
        Args:
            blh_range: (min, max) boundary layer height in meters
            temp_range: (min, max) temperature in Kelvin
            pressure_range: (min, max) surface pressure in hPa
        """
        self.blh_range = blh_range
        self.temp_range = temp_range
        self.pressure_range = pressure_range
        
    def encode_wind_vectors(self, 
                            wind_u: np.ndarray, 
                            wind_v: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode wind vectors as direction + magnitude.
        
        Wind direction is encoded using sin/cos to ensure continuity
        (0° and 360° are treated as the same direction).
        
        Args:
            wind_u: U-component of wind (eastward) [m/s]
            wind_v: V-component of wind (northward) [m/s]
            
        Returns:
            Dictionary with:
            - wind_sin: sine of wind direction
            - wind_cos: cosine of wind direction
            - wind_mag: wind magnitude [m/s]
            - wind_u_norm: normalized U-component
            - wind_v_norm: normalized V-component
        """
        # Compute magnitude
        wind_mag = np.sqrt(wind_u**2 + wind_v**2)
        
        # Compute direction angle (in radians)
        # atan2 returns angle in [-π, π]
        wind_dir = np.arctan2(wind_v, wind_u)
        
        # Encode direction as sin/cos
        wind_sin = np.sin(wind_dir)
        wind_cos = np.cos(wind_dir)
        
        # Normalize U/V by typical max wind speed (~20 m/s)
        max_wind = 20.0
        wind_u_norm = wind_u / max_wind
        wind_v_norm = wind_v / max_wind
        
        # Normalize magnitude
        wind_mag_norm = wind_mag / max_wind
        
        return {
            'wind_sin': wind_sin.astype(np.float32),
            'wind_cos': wind_cos.astype(np.float32),
            'wind_mag': wind_mag_norm.astype(np.float32),
            'wind_u_norm': wind_u_norm.astype(np.float32),
            'wind_v_norm': wind_v_norm.astype(np.float32)
        }
    
    def normalize_blh(self, blh: np.ndarray) -> np.ndarray:
        """
        Normalize boundary layer height to [0, 1] range.
        
        BLH affects NO₂ vertical mixing and surface concentrations.
        Lower BLH leads to higher surface pollution.
        
        Args:
            blh: Boundary layer height in meters
            
        Returns:
            Normalized BLH in [0, 1] range
        """
        blh_min, blh_max = self.blh_range
        normalized = (blh - blh_min) / (blh_max - blh_min)
        return np.clip(normalized, 0, 1).astype(np.float32)
    
    def normalize_temperature(self, temp: np.ndarray) -> np.ndarray:
        """
        Normalize temperature to [0, 1] range.
        
        Args:
            temp: Temperature in Kelvin
            
        Returns:
            Normalized temperature
        """
        t_min, t_max = self.temp_range
        normalized = (temp - t_min) / (t_max - t_min)
        return np.clip(normalized, 0, 1).astype(np.float32)
    
    def normalize_pressure(self, pressure: np.ndarray) -> np.ndarray:
        """
        Normalize surface pressure to [-1, 1] range centered on 1013 hPa.
        
        Args:
            pressure: Surface pressure in hPa
            
        Returns:
            Normalized pressure
        """
        p_min, p_max = self.pressure_range
        p_center = (p_min + p_max) / 2
        p_scale = (p_max - p_min) / 2
        normalized = (pressure - p_center) / p_scale
        return np.clip(normalized, -1, 1).astype(np.float32)
    
    def normalize_humidity(self, humidity: np.ndarray) -> np.ndarray:
        """
        Normalize relative humidity to [0, 1] range.
        
        Args:
            humidity: Relative humidity in %
            
        Returns:
            Normalized humidity
        """
        return np.clip(humidity / 100.0, 0, 1).astype(np.float32)
    
    def encode_temporal_features(self,
                                 day_of_year: int,
                                 hours: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode temporal features with cyclical encoding.
        
        Day-of-year captures seasonal variation in NO₂ (higher in winter).
        Hour-of-day captures diurnal emission patterns.
        
        Args:
            day_of_year: Day of year (1-365)
            hours: Array of hours (0-23)
            
        Returns:
            Dictionary with cyclically encoded temporal features
        """
        # Day of year encoding (seasonal cycle)
        day_angle = 2 * np.pi * day_of_year / 365.0
        day_sin = np.sin(day_angle)
        day_cos = np.cos(day_angle)
        
        # Hour of day encoding (diurnal cycle)
        hour_angle = 2 * np.pi * hours / 24.0
        hour_sin = np.sin(hour_angle)
        hour_cos = np.cos(hour_angle)
        
        # Create arrays matching temporal dimension
        n_times = len(hours)
        
        return {
            'day_sin': np.full(n_times, day_sin, dtype=np.float32),
            'day_cos': np.full(n_times, day_cos, dtype=np.float32),
            'hour_sin': hour_sin.astype(np.float32),
            'hour_cos': hour_cos.astype(np.float32)
        }
    
    def transform(self, aux_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform raw auxiliary data into engineered feature matrix.
        
        Args:
            aux_data: Dictionary with keys:
                - wind_u, wind_v: Wind components
                - blh: Boundary layer height
                - temperature: 2m temperature
                - humidity: Relative humidity
                - pressure: Surface pressure
                - day_of_year: Day number
                - hours: Hour array
                
        Returns:
            Feature matrix of shape (time_steps, n_features)
            Feature order: [wind_u, wind_v, wind_sin, wind_cos, wind_mag,
                           blh, temp, humidity, pressure, 
                           day_sin, day_cos, hour_sin, hour_cos]
        """
        # Encode wind
        wind_features = self.encode_wind_vectors(
            aux_data['wind_u'], aux_data['wind_v']
        )
        
        # Normalize other features
        blh_norm = self.normalize_blh(aux_data['blh'])
        temp_norm = self.normalize_temperature(aux_data['temperature'])
        humidity_norm = self.normalize_humidity(aux_data['humidity'])
        pressure_norm = self.normalize_pressure(aux_data['pressure'])
        
        # Encode temporal features
        temporal_features = self.encode_temporal_features(
            aux_data['day_of_year'], aux_data['hours']
        )
        
        # Stack into feature matrix
        n_times = len(aux_data['hours'])
        features = np.stack([
            wind_features['wind_u_norm'],
            wind_features['wind_v_norm'],
            wind_features['wind_sin'],
            wind_features['wind_cos'],
            wind_features['wind_mag'],
            blh_norm,
            temp_norm,
            humidity_norm,
            pressure_norm,
            temporal_features['day_sin'],
            temporal_features['day_cos'],
            temporal_features['hour_sin'],
            temporal_features['hour_cos']
        ], axis=-1)
        
        return features.astype(np.float32)
    
    @staticmethod
    def get_feature_names() -> list:
        """Get ordered list of feature names."""
        return [
            'wind_u_norm', 'wind_v_norm', 'wind_sin', 'wind_cos', 'wind_mag',
            'blh', 'temperature', 'humidity', 'pressure',
            'day_sin', 'day_cos', 'hour_sin', 'hour_cos'
        ]


class LandUseEmbedding(nn.Module):
    """
    Learnable embeddings for land-use categories.
    
    Instead of one-hot encoding, we use learned embeddings that allow
    the model to discover semantic relationships between land-use types.
    
    Categories:
        0: Water
        1: Rural/Agricultural
        2: Suburban
        3: Urban
        4: Industrial
        5: Forest
        6: Barren
        7: Wetland
        8: Transportation (roads, airports)
        9: Other
    """
    
    def __init__(self, 
                 num_categories: int = 10,
                 embedding_dim: int = 16):
        """
        Initialize land-use embeddings.
        
        Args:
            num_categories: Number of land-use categories
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        # Initialize with small values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
    def forward(self, land_use_map: torch.Tensor) -> torch.Tensor:
        """
        Embed land-use categories.
        
        Args:
            land_use_map: Integer tensor of shape (H, W) with category indices
            
        Returns:
            Embedded features of shape (H, W, embedding_dim)
        """
        return self.embedding(land_use_map)


def broadcast_static_features(static_features: np.ndarray,
                              spatial_shape: Tuple[int, int],
                              mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Broadcast static features across spatial dimensions.
    
    Static features (e.g., land-use embeddings) need to be replicated
    across space while respecting validity masks.
    
    Args:
        static_features: Features of shape (H, W, F) or (F,)
        spatial_shape: Target (H, W)
        mask: Optional validity mask of shape (H, W)
        
    Returns:
        Broadcast features with invalid regions zeroed
    """
    if static_features.ndim == 1:
        # Broadcast single feature vector to all spatial locations
        H, W = spatial_shape
        broadcast = np.broadcast_to(
            static_features[np.newaxis, np.newaxis, :],
            (H, W, len(static_features))
        ).copy()
    else:
        broadcast = static_features.copy()
        
    if mask is not None:
        # Zero out invalid regions
        broadcast = broadcast * mask[:, :, np.newaxis]
        
    return broadcast.astype(np.float32)
