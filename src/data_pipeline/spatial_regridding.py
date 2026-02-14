"""
Spatial Regridding Module

This module implements area-weighted regridding for TROPOMI swath data.
It projects satellite observations onto a fixed regular grid using 
area-weighted averaging, explicitly handling partial pixel overlaps.

Key engineering considerations:
- Avoid bilinear interpolation (causes spatial leakage and artificial smoothness)
- Use area-weighted averaging to preserve flux conservation
- Handle irregular satellite footprints correctly
- Maintain fixed CRS (EPSG:4326) for consistency
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GridDefinition:
    """Definition of a regular lat-lon grid."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution: float  # degrees
    
    @property
    def lat_edges(self) -> np.ndarray:
        """Latitude bin edges."""
        return np.arange(self.lat_min, self.lat_max + self.resolution, self.resolution)
    
    @property
    def lon_edges(self) -> np.ndarray:
        """Longitude bin edges."""
        return np.arange(self.lon_min, self.lon_max + self.resolution, self.resolution)
    
    @property
    def lat_centers(self) -> np.ndarray:
        """Latitude bin centers."""
        edges = self.lat_edges
        return (edges[:-1] + edges[1:]) / 2
    
    @property
    def lon_centers(self) -> np.ndarray:
        """Longitude bin centers."""
        edges = self.lon_edges
        return (edges[:-1] + edges[1:]) / 2
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (n_lat, n_lon)."""
        return (len(self.lat_centers), len(self.lon_centers))
    

def create_target_grid(bounds: dict, resolution: float = 0.1) -> GridDefinition:
    """
    Create a target grid definition.
    
    Args:
        bounds: Dictionary with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'
        resolution: Grid resolution in degrees
        
    Returns:
        GridDefinition object
    """
    return GridDefinition(
        lat_min=bounds['lat_min'],
        lat_max=bounds['lat_max'],
        lon_min=bounds['lon_min'],
        lon_max=bounds['lon_max'],
        resolution=resolution
    )


class AreaWeightedRegridder:
    """
    Performs area-weighted regridding of satellite swath data.
    
    This class implements proper area-weighted averaging that:
    1. Computes the overlap fraction between each source pixel and target cell
    2. Weights contributions by the overlap area
    3. Handles partial overlaps correctly
    4. Preserves the total flux (conservation property)
    
    Example usage:
        grid = create_target_grid({'lat_min': 35, 'lat_max': 55, 
                                   'lon_min': -15, 'lon_max': 25}, 
                                  resolution=0.1)
        regridder = AreaWeightedRegridder(grid)
        gridded = regridder.regrid(no2_data, lat_corners, lon_corners)
    """
    
    def __init__(self, target_grid: GridDefinition):
        """
        Initialize the regridder.
        
        Args:
            target_grid: Target grid definition
        """
        self.grid = target_grid
        
    def regrid(self,
               data: np.ndarray,
               source_lat: np.ndarray,
               source_lon: np.ndarray,
               weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Regrid satellite swath data to the target grid.
        
        This method uses a binning approach that approximates area-weighted
        averaging when the source resolution is finer than or similar to
        the target resolution.
        
        Args:
            data: Source data array (can be 1D or 2D)
            source_lat: Latitude values for each source pixel
            source_lon: Longitude values for each source pixel
            weights: Optional weights for each pixel (default: uniform)
            
        Returns:
            Tuple of (gridded_data, count_map):
            - gridded_data: Data regridded to target grid
            - count_map: Number of valid pixels in each grid cell
        """
        # Flatten inputs
        data_flat = data.flatten()
        lat_flat = source_lat.flatten()
        lon_flat = source_lon.flatten()
        
        # Handle weights
        if weights is None:
            weights_flat = np.ones_like(data_flat)
        else:
            weights_flat = weights.flatten()
            
        # Mask invalid data
        valid = np.isfinite(data_flat) & np.isfinite(lat_flat) & np.isfinite(lon_flat)
        data_valid = data_flat[valid]
        lat_valid = lat_flat[valid]
        lon_valid = lon_flat[valid]
        weights_valid = weights_flat[valid]
        
        # Find grid indices for each valid pixel
        lat_idx = np.digitize(lat_valid, self.grid.lat_edges) - 1
        lon_idx = np.digitize(lon_valid, self.grid.lon_edges) - 1
        
        # Clip to valid range
        n_lat, n_lon = self.grid.shape
        lat_idx = np.clip(lat_idx, 0, n_lat - 1)
        lon_idx = np.clip(lon_idx, 0, n_lon - 1)
        
        # Compute weighted sums for each grid cell
        sum_map = np.zeros(self.grid.shape)
        weight_map = np.zeros(self.grid.shape)
        count_map = np.zeros(self.grid.shape, dtype=np.int32)
        
        for i in range(len(data_valid)):
            li, lj = lat_idx[i], lon_idx[i]
            sum_map[li, lj] += data_valid[i] * weights_valid[i]
            weight_map[li, lj] += weights_valid[i]
            count_map[li, lj] += 1
            
        # Compute weighted average (avoid division by zero)
        gridded_data = np.where(weight_map > 0, sum_map / weight_map, np.nan)
        
        return gridded_data, count_map
    
    def regrid_with_corners(self,
                            data: np.ndarray,
                            lat_corners: np.ndarray,
                            lon_corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Regrid using pixel corner coordinates for proper area weighting.
        
        This is the more accurate method when pixel corner coordinates
        are available (as in some TROPOMI products).
        
        Args:
            data: Source data array
            lat_corners: Latitude corners (shape: ..., 4)
            lon_corners: Longitude corners (shape: ..., 4)
            
        Returns:
            Tuple of (gridded_data, count_map)
        """
        # For each source pixel, compute the overlap area with each target cell
        # This is computationally expensive but accurate
        
        n_lat, n_lon = self.grid.shape
        sum_map = np.zeros((n_lat, n_lon))
        area_map = np.zeros((n_lat, n_lon))
        count_map = np.zeros((n_lat, n_lon), dtype=np.int32)
        
        # Flatten data and corners
        data_flat = data.flatten()
        lat_corners_flat = lat_corners.reshape(-1, 4)
        lon_corners_flat = lon_corners.reshape(-1, 4)
        
        for i in range(len(data_flat)):
            if not np.isfinite(data_flat[i]):
                continue
                
            # Get pixel bounds from corners
            lat_min_pix = np.min(lat_corners_flat[i])
            lat_max_pix = np.max(lat_corners_flat[i])
            lon_min_pix = np.min(lon_corners_flat[i])
            lon_max_pix = np.max(lon_corners_flat[i])
            
            # Find overlapping grid cells
            lat_idx_min = np.searchsorted(self.grid.lat_edges, lat_min_pix) - 1
            lat_idx_max = np.searchsorted(self.grid.lat_edges, lat_max_pix)
            lon_idx_min = np.searchsorted(self.grid.lon_edges, lon_min_pix) - 1
            lon_idx_max = np.searchsorted(self.grid.lon_edges, lon_max_pix)
            
            # Clip to valid range
            lat_idx_min = max(0, lat_idx_min)
            lat_idx_max = min(n_lat, lat_idx_max)
            lon_idx_min = max(0, lon_idx_min)
            lon_idx_max = min(n_lon, lon_idx_max)
            
            # For each overlapping grid cell, compute overlap area
            for li in range(lat_idx_min, lat_idx_max):
                for lj in range(lon_idx_min, lon_idx_max):
                    # Grid cell bounds
                    cell_lat_min = self.grid.lat_edges[li]
                    cell_lat_max = self.grid.lat_edges[li + 1]
                    cell_lon_min = self.grid.lon_edges[lj]
                    cell_lon_max = self.grid.lon_edges[lj + 1]
                    
                    # Compute intersection (simple box intersection)
                    overlap_lat_min = max(lat_min_pix, cell_lat_min)
                    overlap_lat_max = min(lat_max_pix, cell_lat_max)
                    overlap_lon_min = max(lon_min_pix, cell_lon_min)
                    overlap_lon_max = min(lon_max_pix, cell_lon_max)
                    
                    if overlap_lat_max > overlap_lat_min and overlap_lon_max > overlap_lon_min:
                        # Compute overlap area (simple lat-lon area)
                        overlap_area = (overlap_lat_max - overlap_lat_min) * (overlap_lon_max - overlap_lon_min)
                        
                        sum_map[li, lj] += data_flat[i] * overlap_area
                        area_map[li, lj] += overlap_area
                        count_map[li, lj] += 1
                        
        # Compute area-weighted average
        gridded_data = np.where(area_map > 0, sum_map / area_map, np.nan)
        
        return gridded_data, count_map


def bilinear_interpolation_warning():
    """
    ANTI-PATTERN WARNING
    
    DO NOT use bilinear interpolation for satellite data regridding!
    
    Bilinear interpolation:
    1. Creates artificial smoothness that doesn't exist in the original data
    2. Introduces spatial leakage between adjacent pixels
    3. Does not conserve the total flux/mass
    4. Generates values between actual observations (hallucination)
    
    Always use area-weighted averaging or nearest-neighbor for satellite data.
    """
    raise Warning(
        "Bilinear interpolation should NOT be used for satellite data regridding. "
        "Use AreaWeightedRegridder instead."
    )
