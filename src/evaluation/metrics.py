"""
Evaluation Metrics

This module implements comprehensive evaluation metrics for NO₂ prediction,
including spatial, temporal, and seasonal breakdown.

Key engineering considerations:
- Compute RMSE, MAE, R² per region and season
- Evaluate spatial correlation
- Global metrics hide urban failure modes
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class MetricResults:
    """Container for metric results."""
    rmse: float
    mae: float
    r2: float
    bias: float
    spatial_correlation: float
    n_samples: int
    
    def to_dict(self) -> Dict:
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'bias': self.bias,
            'spatial_correlation': self.spatial_correlation,
            'n_samples': self.n_samples
        }


def compute_rmse(predictions: np.ndarray, targets: np.ndarray, 
                 mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Root Mean Square Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional boolean mask (True = valid)
        
    Returns:
        RMSE value
    """
    if mask is not None:
        diff = (predictions - targets)[mask]
    else:
        diff = predictions - targets
        
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_mae(predictions: np.ndarray, targets: np.ndarray,
                mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional boolean mask
        
    Returns:
        MAE value
    """
    if mask is not None:
        diff = np.abs(predictions - targets)[mask]
    else:
        diff = np.abs(predictions - targets)
        
    return float(np.mean(diff))


def compute_r2(predictions: np.ndarray, targets: np.ndarray,
               mask: Optional[np.ndarray] = None) -> float:
    """
    Compute coefficient of determination (R²).
    
    R² = 1 - SS_res / SS_tot
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional boolean mask
        
    Returns:
        R² value (can be negative if model is worse than mean)
    """
    if mask is not None:
        pred = predictions[mask]
        targ = targets[mask]
    else:
        pred = predictions.flatten()
        targ = targets.flatten()
        
    ss_res = np.sum((targ - pred) ** 2)
    ss_tot = np.sum((targ - np.mean(targ)) ** 2)
    
    if ss_tot == 0:
        return 0.0
        
    return float(1 - ss_res / ss_tot)


def compute_bias(predictions: np.ndarray, targets: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> float:
    """
    Compute mean bias (predictions - targets).
    
    Positive bias = overestimation
    Negative bias = underestimation
    """
    if mask is not None:
        diff = (predictions - targets)[mask]
    else:
        diff = predictions - targets
        
    return float(np.mean(diff))


def compute_spatial_correlation(predictions: np.ndarray, 
                                 targets: np.ndarray) -> float:
    """
    Compute spatial correlation coefficient.
    
    Measures how well the spatial pattern of predictions matches targets.
    
    Args:
        predictions: Shape (H, W) or (N, H, W)
        targets: Shape (H, W) or (N, H, W)
        
    Returns:
        Pearson correlation coefficient
    """
    pred_flat = predictions.flatten()
    targ_flat = targets.flatten()
    
    # Remove NaN
    valid = np.isfinite(pred_flat) & np.isfinite(targ_flat)
    pred_flat = pred_flat[valid]
    targ_flat = targ_flat[valid]
    
    if len(pred_flat) < 2:
        return 0.0
        
    correlation = np.corrcoef(pred_flat, targ_flat)[0, 1]
    return float(correlation) if np.isfinite(correlation) else 0.0


def compute_metrics(predictions: np.ndarray,
                    targets: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> MetricResults:
    """
    Compute all standard metrics.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional validity mask
        
    Returns:
        MetricResults with all metrics
    """
    return MetricResults(
        rmse=compute_rmse(predictions, targets, mask),
        mae=compute_mae(predictions, targets, mask),
        r2=compute_r2(predictions, targets, mask),
        bias=compute_bias(predictions, targets, mask),
        spatial_correlation=compute_spatial_correlation(predictions, targets),
        n_samples=int(np.sum(mask)) if mask is not None else predictions.size
    )


class MetricsCalculator:
    """
    Comprehensive metrics calculator with regional and seasonal breakdown.
    
    This class computes metrics stratified by:
    - Region type (urban, rural, industrial, background)
    - Season (winter, spring, summer, autumn)
    - Pollution level (low, medium, high)
    
    This is critical because global metrics can hide systematic failures
    in specific regimes (e.g., underestimating urban peaks).
    """
    
    def __init__(self,
                 region_mask: Optional[np.ndarray] = None,
                 region_labels: Optional[Dict[int, str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            region_mask: Integer mask defining regions, shape (H, W)
            region_labels: Mapping from region ID to name
        """
        self.region_mask = region_mask
        self.region_labels = region_labels or {
            0: 'background',
            1: 'rural',
            2: 'suburban', 
            3: 'urban',
            4: 'industrial'
        }
        
    def compute_all(self,
                    predictions: np.ndarray,
                    targets: np.ndarray,
                    seasons: Optional[np.ndarray] = None,
                    validity_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Compute comprehensive metrics breakdown.
        
        Args:
            predictions: Shape (N, H, W) or (H, W)
            targets: Shape (N, H, W) or (H, W)
            seasons: Optional season indices for each sample (0-3)
            validity_mask: Optional mask for valid pixels
            
        Returns:
            Dictionary with global, regional, and seasonal metrics
        """
        results = {}
        
        # Global metrics
        results['global'] = compute_metrics(predictions, targets, validity_mask).to_dict()
        
        # Regional metrics
        if self.region_mask is not None:
            results['regional'] = {}
            # Only compute regional metrics if shapes match
            if predictions.shape[-2:] == self.region_mask.shape:
                for region_id, region_name in self.region_labels.items():
                    region_mask = self.region_mask == region_id
                    
                    # Broadcast mask to match predictions shape if needed
                    if predictions.ndim == 3:
                        region_mask = np.broadcast_to(region_mask, predictions.shape)
                    
                    # Combine with validity mask if provided
                    if validity_mask is not None:
                        combined_mask = region_mask & validity_mask
                    else:
                        combined_mask = region_mask
                        
                    if np.sum(combined_mask) > 0:
                        results['regional'][region_name] = compute_metrics(
                            predictions, targets, combined_mask
                        ).to_dict()
                    
        # Seasonal metrics
        if seasons is not None:
            season_names = ['winter', 'spring', 'summer', 'autumn']
            results['seasonal'] = {}
            
            for season_id, season_name in enumerate(season_names):
                season_mask = seasons == season_id
                if np.sum(season_mask) > 0:
                    season_preds = predictions[season_mask]
                    season_targets = targets[season_mask]
                    results['seasonal'][season_name] = compute_metrics(
                        season_preds, season_targets
                    ).to_dict()
                    
        # Pollution level breakdown
        results['by_pollution_level'] = self._compute_by_pollution_level(
            predictions, targets
        )
        
        return results
    
    def _compute_by_pollution_level(self,
                                     predictions: np.ndarray,
                                     targets: np.ndarray) -> Dict:
        """
        Compute metrics stratified by pollution level.
        
        This helps identify if the model systematically under/overestimates
        at different pollution levels.
        """
        results = {}
        
        # Define percentile-based thresholds
        valid_targets = targets[np.isfinite(targets)]
        if len(valid_targets) == 0:
            return results
            
        low_threshold = np.percentile(valid_targets, 33)
        high_threshold = np.percentile(valid_targets, 67)
        
        # Low pollution
        low_mask = targets < low_threshold
        if np.sum(low_mask) > 0:
            results['low'] = compute_metrics(predictions, targets, low_mask).to_dict()
            
        # Medium pollution
        medium_mask = (targets >= low_threshold) & (targets < high_threshold)
        if np.sum(medium_mask) > 0:
            results['medium'] = compute_metrics(predictions, targets, medium_mask).to_dict()
            
        # High pollution
        high_mask = targets >= high_threshold
        if np.sum(high_mask) > 0:
            results['high'] = compute_metrics(predictions, targets, high_mask).to_dict()
            
        return results
    
    def create_comparison_table(self, results: Dict) -> str:
        """
        Create a formatted comparison table from results.
        
        Args:
            results: Results dictionary from compute_all
            
        Returns:
            Formatted table string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("EVALUATION METRICS SUMMARY")
        lines.append("=" * 70)
        
        # Global metrics
        lines.append("\nGLOBAL METRICS:")
        lines.append("-" * 40)
        if 'global' in results:
            g = results['global']
            lines.append(f"  RMSE: {g['rmse']:.6f}")
            lines.append(f"  MAE:  {g['mae']:.6f}")
            lines.append(f"  R²:   {g['r2']:.4f}")
            lines.append(f"  Bias: {g['bias']:.6f}")
            lines.append(f"  Spatial Correlation: {g['spatial_correlation']:.4f}")
            
        # Regional breakdown
        if 'regional' in results and results['regional']:
            lines.append("\nREGIONAL BREAKDOWN:")
            lines.append("-" * 40)
            lines.append(f"{'Region':<15} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
            for region, metrics in results['regional'].items():
                lines.append(
                    f"{region:<15} {metrics['rmse']:<12.6f} "
                    f"{metrics['mae']:<12.6f} {metrics['r2']:<10.4f}"
                )
                
        # Seasonal breakdown
        if 'seasonal' in results and results['seasonal']:
            lines.append("\nSEASONAL BREAKDOWN:")
            lines.append("-" * 40)
            lines.append(f"{'Season':<15} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
            for season, metrics in results['seasonal'].items():
                lines.append(
                    f"{season:<15} {metrics['rmse']:<12.6f} "
                    f"{metrics['mae']:<12.6f} {metrics['r2']:<10.4f}"
                )
                
        # Pollution level breakdown
        if 'by_pollution_level' in results and results['by_pollution_level']:
            lines.append("\nBY POLLUTION LEVEL:")
            lines.append("-" * 40)
            lines.append(f"{'Level':<15} {'RMSE':<12} {'MAE':<12} {'Bias':<12}")
            for level, metrics in results['by_pollution_level'].items():
                lines.append(
                    f"{level:<15} {metrics['rmse']:<12.6f} "
                    f"{metrics['mae']:<12.6f} {metrics['bias']:<12.6f}"
                )
                
        lines.append("=" * 70)
        
        return "\n".join(lines)
