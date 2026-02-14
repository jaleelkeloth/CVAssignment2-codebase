"""
Robust Normalization Strategy

This module implements robust scaling using median and IQR (interquartile range)
to stabilize optimization under extreme pollution events.

Key engineering considerations:
- Apply robust scaling (median/IQR) per modality
- Keep satellite and auxiliary normalization INDEPENDENT
- Do NOT normalize across space-time jointly (destroys local structure)
- Store statistics for inverse transformation at inference
- Handle extreme outliers gracefully
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NormalizationStats:
    """Container for normalization statistics."""
    median: float
    iqr: float
    q1: float
    q3: float
    min_val: float
    max_val: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'median': self.median,
            'iqr': self.iqr,
            'q1': self.q1,
            'q3': self.q3,
            'min_val': self.min_val,
            'max_val': self.max_val
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'NormalizationStats':
        """Create from dictionary."""
        return cls(**d)


class RobustNormalizer:
    """
    Robust normalization using median and IQR.
    
    Unlike mean/std normalization, robust scaling is less sensitive to
    extreme outliers (e.g., pollution spikes during wildfires or industrial
    accidents). This is critical for NO₂ data where concentrations can
    vary over several orders of magnitude.
    
    Scaling formula: x_scaled = (x - median) / IQR
    
    Where IQR = Q3 - Q1 (75th percentile - 25th percentile)
    
    Example usage:
        normalizer = RobustNormalizer()
        normalizer.fit(training_data)
        scaled = normalizer.transform(data)
        original = normalizer.inverse_transform(scaled)
    """
    
    def __init__(self, 
                 clip_range: Optional[Tuple[float, float]] = (-10, 10),
                 eps: float = 1e-6):
        """
        Initialize the normalizer.
        
        Args:
            clip_range: Optional (min, max) to clip scaled values. Prevents
                       extreme outliers from dominating gradients.
            eps: Small constant to avoid division by zero for zero-IQR features
        """
        self.clip_range = clip_range
        self.eps = eps
        self.stats: Optional[NormalizationStats] = None
        
    def fit(self, data: np.ndarray) -> 'RobustNormalizer':
        """
        Compute normalization statistics from data.
        
        Args:
            data: Training data array (any shape, stats computed over all elements)
            
        Returns:
            Self for method chaining
        """
        # Flatten and remove NaN
        flat = data.flatten()
        valid = flat[np.isfinite(flat)]
        
        if len(valid) == 0:
            raise ValueError("No valid (finite) values in data for normalization")
            
        # Compute robust statistics
        q1 = np.percentile(valid, 25)
        median = np.percentile(valid, 50)
        q3 = np.percentile(valid, 75)
        iqr = q3 - q1
        
        # Handle zero IQR (constant data)
        if iqr < self.eps:
            iqr = np.std(valid) + self.eps
            
        self.stats = NormalizationStats(
            median=median,
            iqr=iqr,
            q1=q1,
            q3=q3,
            min_val=np.min(valid),
            max_val=np.max(valid)
        )
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply robust scaling to data.
        
        Args:
            data: Data array to transform
            
        Returns:
            Scaled data
        """
        if self.stats is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
            
        scaled = (data - self.stats.median) / self.stats.iqr
        
        if self.clip_range is not None:
            scaled = np.clip(scaled, self.clip_range[0], self.clip_range[1])
            
        return scaled.astype(np.float32)
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling transformation.
        
        Args:
            scaled_data: Scaled data array
            
        Returns:
            Data in original scale
        """
        if self.stats is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
            
        return (scaled_data * self.stats.iqr + self.stats.median).astype(np.float32)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            data: Training data array
            
        Returns:
            Scaled data
        """
        return self.fit(data).transform(data)
    
    def get_stats(self) -> Optional[NormalizationStats]:
        """Get the fitted statistics."""
        return self.stats
    
    def set_stats(self, stats: NormalizationStats):
        """Set statistics (for loading from checkpoint)."""
        self.stats = stats


class PerChannelNormalizer:
    """
    Applies robust normalization independently to each channel/feature.
    
    This is essential for auxiliary features where each variable has
    different physical units and scales (e.g., wind in m/s, temperature
    in K, pressure in hPa).
    """
    
    def __init__(self, 
                 n_channels: int,
                 clip_range: Optional[Tuple[float, float]] = (-10, 10)):
        """
        Initialize per-channel normalizer.
        
        Args:
            n_channels: Number of channels/features
            clip_range: Optional clipping range for scaled values
        """
        self.n_channels = n_channels
        self.normalizers = [
            RobustNormalizer(clip_range=clip_range) 
            for _ in range(n_channels)
        ]
        
    def fit(self, data: np.ndarray, channel_axis: int = -1) -> 'PerChannelNormalizer':
        """
        Fit normalizers for each channel.
        
        Args:
            data: Training data with shape (..., n_channels) if channel_axis=-1
            channel_axis: Axis containing channels
            
        Returns:
            Self for method chaining
        """
        # Move channel axis to last position
        data = np.moveaxis(data, channel_axis, -1)
        
        for i in range(self.n_channels):
            self.normalizers[i].fit(data[..., i])
            
        return self
    
    def transform(self, data: np.ndarray, channel_axis: int = -1) -> np.ndarray:
        """
        Apply per-channel normalization.
        
        Args:
            data: Data to transform
            channel_axis: Axis containing channels
            
        Returns:
            Normalized data
        """
        data = np.moveaxis(data, channel_axis, -1)
        
        transformed = np.zeros_like(data, dtype=np.float32)
        for i in range(self.n_channels):
            transformed[..., i] = self.normalizers[i].transform(data[..., i])
            
        return np.moveaxis(transformed, -1, channel_axis)
    
    def inverse_transform(self, data: np.ndarray, channel_axis: int = -1) -> np.ndarray:
        """
        Reverse per-channel normalization.
        
        Args:
            data: Normalized data
            channel_axis: Axis containing channels
            
        Returns:
            Data in original scale
        """
        data = np.moveaxis(data, channel_axis, -1)
        
        original = np.zeros_like(data, dtype=np.float32)
        for i in range(self.n_channels):
            original[..., i] = self.normalizers[i].inverse_transform(data[..., i])
            
        return np.moveaxis(original, -1, channel_axis)
    
    def get_all_stats(self) -> list:
        """Get statistics for all channels."""
        return [n.get_stats() for n in self.normalizers]


def compute_robust_stats(data: np.ndarray, 
                         axis: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute median and IQR along specified axis.
    
    Args:
        data: Input data array
        axis: Axis along which to compute stats (None = all)
        
    Returns:
        Tuple of (median, iqr)
    """
    q1 = np.nanpercentile(data, 25, axis=axis)
    median = np.nanpercentile(data, 50, axis=axis)
    q3 = np.nanpercentile(data, 75, axis=axis)
    iqr = q3 - q1
    
    return median, iqr


def standardize_warning():
    """
    ANTI-PATTERN WARNING
    
    Do NOT use simple mean/std standardization for NO₂ data!
    
    Problems with z-score standardization on pollution data:
    1. Extreme events (wildfires, industrial accidents) dominate statistics
    2. Mean is not representative of typical conditions
    3. Standard deviation is inflated by outliers
    4. Resulting scaled values have poor numerical properties
    
    Use RobustNormalizer with median/IQR instead.
    """
    raise Warning(
        "Mean/std standardization is NOT recommended for NO₂ data. "
        "Use RobustNormalizer instead."
    )


def joint_normalization_warning():
    """
    ANTI-PATTERN WARNING
    
    Do NOT normalize across space and time jointly!
    
    Problems:
    1. Destroys local spatial structure (urban vs rural differences)
    2. Removes seasonal signals (winter vs summer NO₂)
    3. Mixes physical regimes inappropriately
    4. Makes cross-modal attention meaningless
    
    Normalize per-modality, with fixed spatial and temporal structure.
    """
    raise Warning(
        "Joint space-time normalization is NOT recommended. "
        "Normalize per-modality with fixed structure."
    )
