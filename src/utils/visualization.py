"""
Visualization Module

Functions for visualizing:
- Attention maps
- Predictions vs ground truth
- Training curves
- Spatial patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Optional, Tuple, List
from pathlib import Path


class Visualizer:
    """
    Visualization helper for Cross-Modal NO₂ Predictor.
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a nice color scheme
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_training_curves(self, 
                              history: Dict,
                              save_name: str = 'training_curves.png') -> str:
        """
        Plot training and validation loss curves.
        
        Args:
            history: Training history with 'train_loss' and 'val_loss'
            save_name: Filename for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        
        if history.get('val_loss'):
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (Huber)', fontsize=12)
        ax.set_title('Training Curves - Cross-Modal NO₂ Predictor', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Mark warmup/finetune boundary if visible
        warmup_epochs = 10  # Default from config
        if len(epochs) > warmup_epochs:
            ax.axvline(x=warmup_epochs, color='gray', linestyle='--', 
                       alpha=0.5, label='Warmup→Finetune')
            ax.text(warmup_epochs + 0.5, ax.get_ylim()[1] * 0.9, 
                   'Fine-tuning', fontsize=10, color='gray')
            ax.text(warmup_epochs - 4, ax.get_ylim()[1] * 0.9, 
                   'Warmup', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_predictions_vs_actual(self,
                                    predictions: np.ndarray,
                                    targets: np.ndarray,
                                    save_name: str = 'predictions_vs_actual.png') -> str:
        """
        Create scatter plot of predictions vs actual values.
        
        Args:
            predictions: Predicted NO₂ values
            targets: Ground truth NO₂ values
            save_name: Filename for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        pred_flat = predictions.flatten()
        targ_flat = targets.flatten()
        
        # Sample if too many points
        if len(pred_flat) > 10000:
            idx = np.random.choice(len(pred_flat), 10000, replace=False)
            pred_flat = pred_flat[idx]
            targ_flat = targ_flat[idx]
        
        ax.scatter(targ_flat, pred_flat, alpha=0.3, s=10, c='blue', edgecolors='none')
        
        # Perfect prediction line
        min_val = min(targ_flat.min(), pred_flat.min())
        max_val = max(targ_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
        
        # Compute R²
        ss_res = np.sum((targ_flat - pred_flat) ** 2)
        ss_tot = np.sum((targ_flat - np.mean(targ_flat)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        ax.set_xlabel('Actual NO₂ [mol/m²]', fontsize=12)
        ax.set_ylabel('Predicted NO₂ [mol/m²]', fontsize=12)
        ax.set_title(f'Predictions vs Actual (R² = {r2:.4f})', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_spatial_comparison(self,
                                 prediction: np.ndarray,
                                 target: np.ndarray,
                                 save_name: str = 'spatial_comparison.png') -> str:
        """
        Plot side-by-side spatial comparison of prediction and target.
        
        Args:
            prediction: Single prediction field (H, W)
            target: Single target field (H, W)
            save_name: Filename for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        vmin = min(prediction.min(), target.min())
        vmax = max(prediction.max(), target.max())
        
        # Target
        im0 = axes[0].imshow(target, cmap='YlOrRd', vmin=vmin, vmax=vmax)
        axes[0].set_title('Ground Truth NO₂', fontsize=12)
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im0, ax=axes[0], label='mol/m²')
        
        # Prediction
        im1 = axes[1].imshow(prediction, cmap='YlOrRd', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted NO₂', fontsize=12)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[1], label='mol/m²')
        
        # Difference
        diff = prediction - target
        max_diff = max(abs(diff.min()), abs(diff.max()))
        im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
        axes[2].set_title('Prediction - Target', fontsize=12)
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[2], label='Δ mol/m²')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_attention_map(self,
                           attention_weights: np.ndarray,
                           spatial_shape: Tuple[int, int],
                           aux_labels: Optional[List[str]] = None,
                           save_name: str = 'attention_maps.png') -> str:
        """
        Visualize cross-modal attention patterns.
        
        Args:
            attention_weights: Shape (heads, H*W, T) or (H*W, T)
            spatial_shape: (H, W)
            aux_labels: Labels for auxiliary timesteps
            save_name: Filename for saved figure
            
        Returns:
            Path to saved figure
        """
        if attention_weights.ndim == 3:
            # Average over heads
            attn = attention_weights.mean(axis=0)
        else:
            attn = attention_weights
            
        H, W = spatial_shape
        n_spatial, n_aux = attn.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Attention for different spatial locations
        locations = [
            (H//4, W//4, 'Top-Left'),
            (H//4, 3*W//4, 'Top-Right'),
            (3*H//4, W//4, 'Bottom-Left'),
            (3*H//4, 3*W//4, 'Bottom-Right')
        ]
        
        for ax, (y, x, title) in zip(axes.flatten(), locations):
            spatial_idx = y * W + x
            ax.bar(range(n_aux), attn[spatial_idx], color='steelblue', alpha=0.7)
            ax.set_xlabel('Auxiliary Time Step')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'Attention at {title} ({y},{x})')
            ax.set_ylim(0, attn.max() * 1.1)
            
            if aux_labels and len(aux_labels) <= 12:
                ax.set_xticks(range(n_aux))
                ax.set_xticklabels(aux_labels, rotation=45, ha='right')
        
        plt.suptitle('Cross-Modal Attention Patterns', fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_metrics_summary(self,
                              results: Dict,
                              save_name: str = 'metrics_summary.png') -> str:
        """
        Create visual summary of evaluation metrics.
        
        Args:
            results: Results dictionary from MetricsCalculator
            save_name: Filename for saved figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Global metrics bar chart
        ax = axes[0]
        if 'global' in results:
            metrics = results['global']
            names = ['RMSE', 'MAE', 'R²']
            values = [metrics['rmse'] * 1e5, metrics['mae'] * 1e5, metrics['r2']]  # Scale for visibility
            colors = ['#2ecc71', '#3498db', '#9b59b6']
            
            bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.2)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title('Global Metrics', fontsize=12)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Regional breakdown
        ax = axes[1]
        if 'regional' in results and results['regional']:
            regions = list(results['regional'].keys())
            rmse_vals = [results['regional'][r]['rmse'] * 1e5 for r in regions]
            
            bars = ax.bar(regions, rmse_vals, color='steelblue', edgecolor='black')
            ax.set_ylabel('RMSE (×10⁻⁵ mol/m²)', fontsize=11)
            ax.set_title('RMSE by Region', fontsize=12)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
