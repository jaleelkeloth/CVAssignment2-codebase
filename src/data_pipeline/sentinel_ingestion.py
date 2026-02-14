"""
Sentinel-5P TROPOMI NO₂ Data Ingestion Pipeline

This module handles the ingestion of Level-2 TROPOMI NO₂ data from Sentinel-5P satellite.
It implements QA filtering, orbit metadata parsing, and data extraction following best practices
to avoid label leakage and ensure proper masking before any aggregation.

Key engineering considerations:
- QA threshold filtering BEFORE any aggregation
- Preserve orbit metadata for spatial diagnostics
- Never interpolate before masking
- Handle NaN values explicitly

For real data usage:
- Register at Copernicus Data Space: https://dataspace.copernicus.eu/
- Use sentinelsat or hda API to download L2__NO2___ products
"""

import numpy as np
import xarray as xr
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OrbitMetadata:
    """Container for TROPOMI orbit metadata."""
    orbit_number: int
    sensing_start: str
    sensing_end: str
    processing_mode: str  # NRTI, OFFL, RPRO
    processor_version: str
    platform: str
    
    
@dataclass
class NO2DataProduct:
    """Container for processed NO₂ data product."""
    no2_column: np.ndarray  # Tropospheric NO₂ column [mol/m²]
    latitude: np.ndarray
    longitude: np.ndarray
    qa_value: np.ndarray
    time: np.ndarray
    cloud_fraction: Optional[np.ndarray] = None
    amf_trop: Optional[np.ndarray] = None  # Air mass factor
    orbit_metadata: Optional[OrbitMetadata] = None


class SentinelDataLoader:
    """
    Loads and processes Sentinel-5P TROPOMI NO₂ data.
    
    This class handles:
    1. Loading NetCDF4 files from TROPOMI L2 products
    2. Extracting relevant variables (NO₂ column, QA, coordinates)
    3. Parsing orbit metadata
    4. Applying QA filtering
    
    Example usage:
        loader = SentinelDataLoader(qa_threshold=0.75)
        data = loader.load_tropomi_no2('S5P_OFFL_L2__NO2____20230101T*.nc')
        filtered_data = loader.apply_filtering(data)
    """
    
    def __init__(self, qa_threshold: float = 0.75):
        """
        Initialize the data loader.
        
        Args:
            qa_threshold: Quality assurance threshold (0-1). Pixels with 
                         qa_value < threshold are masked. Default 0.75 follows
                         TROPOMI recommendations for scientific analysis.
        """
        self.qa_threshold = qa_threshold
        
    def load_tropomi_no2(self, filepath: str) -> NO2DataProduct:
        """
        Load TROPOMI NO₂ Level-2 NetCDF file.
        
        Args:
            filepath: Path to the NetCDF file (.nc)
            
        Returns:
            NO2DataProduct containing extracted data
            
        Note:
            TROPOMI L2 files have the structure:
            - PRODUCT group: main data fields
            - METADATA group: acquisition metadata
        """
        # Open dataset with xarray, specifying the PRODUCT group
        ds = xr.open_dataset(filepath, group='PRODUCT')
        
        # Extract NO₂ tropospheric column
        # Variable: nitrogendioxide_tropospheric_column
        # Units: mol/m² (needs conversion factor from attributes)
        no2_column = ds['nitrogendioxide_tropospheric_column'].values
        
        # Extract QA value (0-1, higher is better)
        qa_value = ds['qa_value'].values
        
        # Extract coordinates
        latitude = ds['latitude'].values
        longitude = ds['longitude'].values
        
        # Extract time
        time = ds['time'].values
        
        # Optional: cloud fraction
        cloud_fraction = None
        if 'cloud_fraction' in ds:
            cloud_fraction = ds['cloud_fraction'].values
            
        # Optional: air mass factor
        amf_trop = None
        if 'air_mass_factor_troposphere' in ds:
            amf_trop = ds['air_mass_factor_troposphere'].values
            
        ds.close()
        
        # Parse orbit metadata from the METADATA group
        orbit_metadata = self._parse_orbit_metadata(filepath)
        
        return NO2DataProduct(
            no2_column=no2_column,
            latitude=latitude,
            longitude=longitude,
            qa_value=qa_value,
            time=time,
            cloud_fraction=cloud_fraction,
            amf_trop=amf_trop,
            orbit_metadata=orbit_metadata
        )
    
    def _parse_orbit_metadata(self, filepath: str) -> Optional[OrbitMetadata]:
        """
        Parse orbit metadata from the METADATA group.
        
        Args:
            filepath: Path to the NetCDF file
            
        Returns:
            OrbitMetadata object or None if metadata unavailable
        """
        try:
            ds_meta = xr.open_dataset(filepath, group='METADATA/EOP_METADATA')
            
            orbit_metadata = OrbitMetadata(
                orbit_number=int(ds_meta.attrs.get('orbit', 0)),
                sensing_start=str(ds_meta.attrs.get('time_coverage_start', '')),
                sensing_end=str(ds_meta.attrs.get('time_coverage_end', '')),
                processing_mode=str(ds_meta.attrs.get('processing_mode', 'UNKNOWN')),
                processor_version=str(ds_meta.attrs.get('processor_version', '')),
                platform=str(ds_meta.attrs.get('platform', 'Sentinel-5P'))
            )
            ds_meta.close()
            return orbit_metadata
        except Exception:
            # Metadata parsing failed, return None
            return None
        
    def apply_filtering(self, data: NO2DataProduct) -> NO2DataProduct:
        """
        Apply QA filtering to the NO₂ data.
        
        This method applies the QA threshold mask to the NO₂ column data.
        Pixels with qa_value < threshold are set to NaN.
        
        CRITICAL: This must be done BEFORE any spatial aggregation or 
        interpolation to avoid contaminating good pixels with bad data.
        
        Args:
            data: NO2DataProduct to filter
            
        Returns:
            Filtered NO2DataProduct with masked values as NaN
        """
        # Create mask for valid pixels
        valid_mask = data.qa_value >= self.qa_threshold
        
        # Apply mask to NO₂ column
        filtered_no2 = np.where(valid_mask, data.no2_column, np.nan)
        
        # Also filter cloud fraction if available
        filtered_cloud = None
        if data.cloud_fraction is not None:
            filtered_cloud = np.where(valid_mask, data.cloud_fraction, np.nan)
            
        return NO2DataProduct(
            no2_column=filtered_no2,
            latitude=data.latitude,
            longitude=data.longitude,
            qa_value=data.qa_value,
            time=data.time,
            cloud_fraction=filtered_cloud,
            amf_trop=data.amf_trop,
            orbit_metadata=data.orbit_metadata
        )


def apply_qa_filtering(no2_data: np.ndarray, 
                       qa_values: np.ndarray, 
                       threshold: float = 0.75) -> np.ndarray:
    """
    Standalone function to apply QA filtering to NO₂ data.
    
    Args:
        no2_data: NO₂ column data array
        qa_values: Quality assurance values (0-1)
        threshold: Minimum QA value to accept
        
    Returns:
        Filtered NO₂ data with low-QA pixels masked as NaN
    """
    return np.where(qa_values >= threshold, no2_data, np.nan)


class SyntheticDataGenerator:
    """
    Generates synthetic NO₂ data for demonstration and testing purposes.
    
    This generator creates realistic-looking NO₂ patterns including:
    - Urban hotspots with elevated NO₂
    - Rural background levels
    - Diurnal variation
    - Seasonal patterns
    - Wind-driven transport effects
    - Realistic noise characteristics
    
    The synthetic data mimics the statistical properties of real TROPOMI
    observations while being fully deterministic and reproducible.
    """
    
    def __init__(self, 
                 grid_height: int = 64,
                 grid_width: int = 64,
                 time_steps: int = 24,
                 seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            grid_height: Height of spatial grid
            grid_width: Width of spatial grid
            time_steps: Number of temporal steps per sample
            seed: Random seed for reproducibility
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.time_steps = time_steps
        self.rng = np.random.default_rng(seed)
        
        # Pre-compute static features
        self._init_static_features()
        
    def _init_static_features(self):
        """Initialize static spatial features (urban centers, terrain)."""
        # Create urban center locations (fixed sources)
        self.urban_centers = [
            (self.grid_height // 3, self.grid_width // 4),
            (self.grid_height // 2, self.grid_width // 2),
            (2 * self.grid_height // 3, 3 * self.grid_width // 4),
        ]
        
        # Create land use map (0: water, 1: rural, 2: suburban, 3: urban, 4: industrial)
        self.land_use = np.ones((self.grid_height, self.grid_width), dtype=np.int32)
        for cy, cx in self.urban_centers:
            # Create urban cores with industrial zones
            yy, xx = np.ogrid[:self.grid_height, :self.grid_width]
            dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            self.land_use = np.where(dist < 5, 3, self.land_use)  # Urban core
            self.land_use = np.where((dist >= 5) & (dist < 10), 2, self.land_use)  # Suburban
            
    def generate_auxiliary_features(self) -> Dict[str, np.ndarray]:
        """
        Generate synthetic auxiliary meteorological features.
        
        Returns:
            Dictionary containing:
            - wind_u: U-component of wind [m/s]
            - wind_v: V-component of wind [m/s]
            - blh: Boundary layer height [m]
            - temperature: 2m temperature [K]
            - humidity: Relative humidity [%]
            - pressure: Surface pressure [hPa]
            - time_features: (day_of_year, hour) normalized
        """
        # Wind patterns with temporal variation
        base_u = 3.0 * np.sin(np.linspace(0, 2*np.pi, self.time_steps))
        base_v = 2.0 * np.cos(np.linspace(0, 2*np.pi, self.time_steps))
        
        wind_u = base_u + self.rng.normal(0, 0.5, self.time_steps)
        wind_v = base_v + self.rng.normal(0, 0.5, self.time_steps)
        
        # Boundary layer height (diurnal cycle)
        blh_base = 500 + 1000 * np.sin(np.linspace(0, np.pi, self.time_steps))
        blh = blh_base + self.rng.normal(0, 50, self.time_steps)
        blh = np.clip(blh, 100, 3000)
        
        # Temperature (diurnal cycle)
        temp_base = 288 + 5 * np.sin(np.linspace(-np.pi/2, np.pi/2, self.time_steps))
        temperature = temp_base + self.rng.normal(0, 1, self.time_steps)
        
        # Humidity (inverse of temperature)
        humidity = 60 - 10 * np.sin(np.linspace(-np.pi/2, np.pi/2, self.time_steps))
        humidity = humidity + self.rng.normal(0, 5, self.time_steps)
        humidity = np.clip(humidity, 20, 100)
        
        # Surface pressure
        pressure = 1013 + self.rng.normal(0, 5, self.time_steps)
        
        # Time features (normalized)
        day_of_year = self.rng.integers(1, 366)
        hours = np.arange(self.time_steps) % 24
        
        return {
            'wind_u': wind_u.astype(np.float32),
            'wind_v': wind_v.astype(np.float32),
            'blh': blh.astype(np.float32),
            'temperature': temperature.astype(np.float32),
            'humidity': humidity.astype(np.float32),
            'pressure': pressure.astype(np.float32),
            'day_of_year': day_of_year,
            'hours': hours.astype(np.float32)
        }
        
    def generate_no2_field(self, 
                           aux_features: Dict[str, np.ndarray],
                           noise_level: float = 0.1) -> np.ndarray:
        """
        Generate synthetic NO₂ field based on auxiliary features.
        
        This creates physically-motivated NO₂ patterns:
        - Urban emissions as Gaussian plumes
        - Wind-driven transport (advection)
        - BLH-dependent dilution
        - Diurnal emission pattern
        
        Args:
            aux_features: Auxiliary meteorological features
            noise_level: Relative noise level (0-1)
            
        Returns:
            NO₂ field with shape (time_steps, grid_height, grid_width)
        """
        no2_field = np.zeros((self.time_steps, self.grid_height, self.grid_width))
        
        # Create coordinate grids
        yy, xx = np.meshgrid(
            np.arange(self.grid_height), 
            np.arange(self.grid_width), 
            indexing='ij'
        )
        
        for t in range(self.time_steps):
            # Diurnal emission pattern (higher during rush hours)
            hour = aux_features['hours'][t]
            emission_factor = 1.0 + 0.5 * np.exp(-((hour - 8)**2 + (hour - 18)**2) / 20)
            
            # BLH dilution effect (lower BLH = higher surface concentration)
            blh = aux_features['blh'][t]
            dilution_factor = 1000.0 / blh
            
            # Wind transport effect
            wind_u = aux_features['wind_u'][t]
            wind_v = aux_features['wind_v'][t]
            
            # Generate NO₂ from each urban source
            frame = np.zeros((self.grid_height, self.grid_width))
            
            for cy, cx in self.urban_centers:
                # Base emission strength (varies by source)
                emission = 8e-5 * emission_factor * dilution_factor
                
                # Gaussian plume with wind advection
                # Plume spreads more in downwind direction
                wind_speed = np.sqrt(wind_u**2 + wind_v**2) + 0.1
                wind_dir_x = wind_u / wind_speed
                wind_dir_y = wind_v / wind_speed
                
                # Distance from source
                dx = xx - cx
                dy = yy - cy
                dist = np.sqrt(dx**2 + dy**2)
                
                # Downwind distance
                downwind = dx * wind_dir_x + dy * wind_dir_y
                crosswind = abs(dx * (-wind_dir_y) + dy * wind_dir_x)
                
                # Gaussian plume model
                sigma_y = 2.0 + 0.2 * np.maximum(downwind, 0)
                sigma_z = 2.0 + 0.1 * np.maximum(downwind, 0)
                
                plume = emission * np.exp(
                    -crosswind**2 / (2 * sigma_y**2)
                ) * np.exp(
                    -dist**2 / (2 * (sigma_y * sigma_z))
                )
                
                # Only add positive contributions (downwind)
                plume = np.where(downwind > -5, plume, plume * 0.1)
                frame += plume
                
            # Add background NO₂ (rural levels)
            background = 2e-5 + self.rng.normal(0, 1e-6, frame.shape)
            frame += np.abs(background)
            
            # Add noise
            noise = self.rng.normal(0, noise_level * np.mean(frame), frame.shape)
            frame = frame + noise
            frame = np.maximum(frame, 0)  # NO₂ cannot be negative
            
            no2_field[t] = frame
            
        return no2_field.astype(np.float32)
    
    def generate_sample(self, noise_level: float = 0.1) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """
        Generate a complete synthetic sample.
        
        Returns:
            Tuple of (no2_field, aux_features, land_use):
            - no2_field: (time_steps, H, W) NO₂ concentrations
            - aux_features: Dictionary of meteorological features
            - land_use: (H, W) land use categories
        """
        aux_features = self.generate_auxiliary_features()
        no2_field = self.generate_no2_field(aux_features, noise_level)
        
        return no2_field, aux_features, self.land_use.copy()
    
    def generate_dataset(self, 
                         num_samples: int,
                         noise_level: float = 0.1) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            noise_level: Relative noise level
            
        Returns:
            Tuple of (no2_fields, aux_features_list, land_use):
            - no2_fields: (N, time_steps, H, W) NO₂ concentrations
            - aux_features_list: List of N auxiliary feature dictionaries
            - land_use: (H, W) land use categories (shared)
        """
        no2_fields = []
        aux_features_list = []
        
        for i in range(num_samples):
            # Vary the seed for each sample while maintaining reproducibility
            self.rng = np.random.default_rng(42 + i)
            no2, aux, _ = self.generate_sample(noise_level)
            no2_fields.append(no2)
            aux_features_list.append(aux)
            
        no2_fields = np.stack(no2_fields, axis=0)
        
        return no2_fields, aux_features_list, self.land_use.copy()
