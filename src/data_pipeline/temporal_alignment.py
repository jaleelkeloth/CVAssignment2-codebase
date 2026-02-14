"""
Temporal Alignment Engine

This module synchronizes satellite data with auxiliary meteorological data streams.
It aligns observations to satellite overpass times using appropriate methods
to preserve the physical meaning of meteorological drivers.

Key engineering considerations:
- Misaligned time windows destroy cross-attention interpretability
- Use nearest-neighbor or windowed aggregation, NOT linear interpolation
- Validate temporal consistency between data streams
- Handle missing timestamps gracefully
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class TemporalWindow:
    """Specification for temporal aggregation window."""
    center_time: datetime
    window_hours: float
    method: str  # 'nearest', 'mean', 'max', 'min'
    

class TemporalAligner:
    """
    Aligns auxiliary data streams to satellite overpass times.
    
    Sentinel-5P has a sun-synchronous orbit with local equator crossing 
    time around 13:30. Meteorological data (e.g., ERA5) typically comes
    at hourly resolution. This class aligns auxiliary data to the 
    satellite overpass time using appropriate aggregation methods.
    
    Example usage:
        aligner = TemporalAligner(window_hours=1.0)
        aligned_aux = aligner.align_to_overpass(
            aux_data={'wind_u': era5_u, 'wind_v': era5_v},
            aux_times=era5_times,
            overpass_time=satellite_time
        )
    """
    
    def __init__(self, 
                 window_hours: float = 1.0,
                 method: str = 'nearest'):
        """
        Initialize the temporal aligner.
        
        Args:
            window_hours: Size of temporal window for aggregation (hours)
            method: Aggregation method ('nearest', 'mean', 'max', 'min')
        """
        self.window_hours = window_hours
        self.method = method
        
        # Validate method
        valid_methods = ['nearest', 'mean', 'max', 'min']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
    def align_to_overpass(self,
                          aux_data: Dict[str, np.ndarray],
                          aux_times: np.ndarray,
                          overpass_time: Union[datetime, np.datetime64]) -> Dict[str, np.ndarray]:
        """
        Align auxiliary data to satellite overpass time.
        
        Args:
            aux_data: Dictionary of auxiliary variables, each with shape (T, ...)
            aux_times: Array of timestamps for auxiliary data
            overpass_time: Target satellite overpass time
            
        Returns:
            Dictionary of aligned auxiliary data at overpass time
        """
        # Convert overpass time to numpy datetime64 if needed
        if isinstance(overpass_time, datetime):
            overpass_time = np.datetime64(overpass_time)
            
        # Find indices within the temporal window
        window_delta = np.timedelta64(int(self.window_hours * 60), 'm')
        time_diffs = np.abs(aux_times - overpass_time)
        
        if self.method == 'nearest':
            # Find the nearest timestamp
            idx = np.argmin(time_diffs)
            aligned = {k: v[idx] for k, v in aux_data.items()}
            
        else:
            # Find all timestamps within window
            in_window = time_diffs <= window_delta
            
            if not np.any(in_window):
                # No data in window, fall back to nearest
                idx = np.argmin(time_diffs)
                aligned = {k: v[idx] for k, v in aux_data.items()}
            else:
                # Aggregate within window
                aligned = {}
                for key, values in aux_data.items():
                    window_values = values[in_window]
                    
                    if self.method == 'mean':
                        aligned[key] = np.nanmean(window_values, axis=0)
                    elif self.method == 'max':
                        aligned[key] = np.nanmax(window_values, axis=0)
                    elif self.method == 'min':
                        aligned[key] = np.nanmin(window_values, axis=0)
                        
        return aligned
    
    def windowed_aggregation(self,
                             data: np.ndarray,
                             data_times: np.ndarray,
                             center_time: Union[datetime, np.datetime64],
                             window_hours: Optional[float] = None) -> np.ndarray:
        """
        Aggregate data within a temporal window around center time.
        
        Args:
            data: Data array with shape (T, ...)
            data_times: Array of timestamps
            center_time: Center of aggregation window
            window_hours: Window size (uses instance default if None)
            
        Returns:
            Aggregated data
        """
        if window_hours is None:
            window_hours = self.window_hours
            
        if isinstance(center_time, datetime):
            center_time = np.datetime64(center_time)
            
        window_delta = np.timedelta64(int(window_hours * 60), 'm')
        time_diffs = np.abs(data_times - center_time)
        in_window = time_diffs <= window_delta
        
        if not np.any(in_window):
            # Return nearest value if window is empty
            idx = np.argmin(time_diffs)
            return data[idx]
            
        window_data = data[in_window]
        
        if self.method == 'mean':
            return np.nanmean(window_data, axis=0)
        elif self.method == 'max':
            return np.nanmax(window_data, axis=0)
        elif self.method == 'min':
            return np.nanmin(window_data, axis=0)
        else:  # nearest
            time_diff_in_window = time_diffs[in_window]
            nearest_idx = np.argmin(time_diff_in_window)
            return window_data[nearest_idx]
    
    def validate_temporal_consistency(self,
                                      sat_times: np.ndarray,
                                      aux_times: np.ndarray,
                                      max_gap_hours: float = 3.0) -> Tuple[bool, Dict]:
        """
        Validate temporal consistency between satellite and auxiliary data.
        
        Checks that auxiliary data coverage is sufficient for all satellite
        observations and identifies any problematic gaps.
        
        Args:
            sat_times: Array of satellite observation times
            aux_times: Array of auxiliary data times
            max_gap_hours: Maximum acceptable gap in hours
            
        Returns:
            Tuple of (is_valid, diagnostic_info):
            - is_valid: True if temporal alignment is feasible
            - diagnostic_info: Dictionary with gap statistics
        """
        max_gap_delta = np.timedelta64(int(max_gap_hours * 60), 'm')
        
        # For each satellite time, find the nearest auxiliary time
        gaps = []
        problematic_times = []
        
        aux_times_sorted = np.sort(aux_times)
        
        for sat_time in sat_times:
            # Find nearest auxiliary time
            time_diffs = np.abs(aux_times_sorted - sat_time)
            min_gap = np.min(time_diffs)
            gaps.append(min_gap)
            
            if min_gap > max_gap_delta:
                problematic_times.append(sat_time)
                
        gaps = np.array(gaps)
        
        diagnostic_info = {
            'min_gap': gaps.min() if len(gaps) > 0 else None,
            'max_gap': gaps.max() if len(gaps) > 0 else None,
            'mean_gap': np.mean(gaps) if len(gaps) > 0 else None,
            'num_problematic': len(problematic_times),
            'problematic_times': problematic_times[:10],  # First 10
            'coverage_fraction': 1.0 - len(problematic_times) / len(sat_times) if len(sat_times) > 0 else 0
        }
        
        is_valid = len(problematic_times) == 0
        
        return is_valid, diagnostic_info
    
    
def create_time_series(start_time: datetime,
                       num_steps: int,
                       step_hours: float = 1.0) -> np.ndarray:
    """
    Create a regular time series.
    
    Args:
        start_time: Start datetime
        num_steps: Number of time steps
        step_hours: Time step in hours
        
    Returns:
        Array of datetime64 timestamps
    """
    start = np.datetime64(start_time)
    step = np.timedelta64(int(step_hours * 60), 'm')
    return np.array([start + i * step for i in range(num_steps)])


def interpolation_warning():
    """
    ANTI-PATTERN WARNING
    
    DO NOT use linear interpolation for temporal alignment of meteorological data
    with satellite observations!
    
    Linear interpolation:
    1. Creates fictional intermediate states that were never observed
    2. Smooths out rapid weather changes that affect NO₂ transport
    3. Destroys the physical meaning of instantaneous observations
    4. Makes cross-attention weights uninterpretable
    
    Use nearest-neighbor or windowed aggregation instead.
    """
    raise Warning(
        "Linear temporal interpolation should NOT be used for satellite-auxiliary alignment. "
        "Use TemporalAligner with 'nearest' or 'mean' method instead."
    )
