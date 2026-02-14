"""
Robustness Stress Testing

This module validates model behavior under missing or noisy inputs.

Key engineering considerations:
- Randomly drop satellite pixels during inference
- Drop auxiliary channels to test sensitivity
- Cross-modal attention should reweight, not fail
- Graceful degradation is expected
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from .metrics import compute_metrics, MetricResults


class RobustnessTester:
    """
    Tests model robustness under various input perturbations.
    
    A robust cross-modal attention model should:
    1. Gracefully degrade when satellite data is partially missing
    2. Reweight attention when auxiliary channels are dropped
    3. Not catastrophically fail under any reasonable perturbation
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize robustness tester.
        
        Args:
            model: Trained model to test
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def test_satellite_dropout(self,
                                satellite_data: torch.Tensor,
                                auxiliary_data: torch.Tensor,
                                targets: torch.Tensor,
                                drop_probs: List[float] = [0.1, 0.2, 0.3, 0.5]) -> Dict:
        """
        Test model under satellite pixel dropout.
        
        Simulates missing satellite observations (cloud cover, sensor gaps).
        
        Args:
            satellite_data: Shape (B, C, H, W)
            auxiliary_data: Shape (B, T, F)
            targets: Shape (B, H, W)
            drop_probs: List of dropout probabilities to test
            
        Returns:
            Dictionary with metrics for each dropout level
        """
        results = {}
        
        # Baseline (no dropout)
        with torch.no_grad():
            baseline_pred, _ = self.model(
                satellite_data.to(self.device),
                auxiliary_data.to(self.device)
            )
        baseline_metrics = compute_metrics(
            baseline_pred.cpu().numpy(),
            targets.numpy()
        )
        results['baseline'] = baseline_metrics.to_dict()
        
        # Test each dropout level
        for drop_prob in drop_probs:
            # Create dropout mask
            mask = torch.rand_like(satellite_data) > drop_prob
            dropped_satellite = satellite_data * mask.float()
            
            # Fill dropped values with 0 (or could use mean)
            
            with torch.no_grad():
                pred, _ = self.model(
                    dropped_satellite.to(self.device),
                    auxiliary_data.to(self.device)
                )
                
            metrics = compute_metrics(pred.cpu().numpy(), targets.numpy())
            results[f'drop_{int(drop_prob*100)}pct'] = metrics.to_dict()
            
            # Compute degradation ratio
            results[f'drop_{int(drop_prob*100)}pct']['rmse_degradation'] = (
                metrics.rmse / baseline_metrics.rmse - 1
            ) * 100
            
        return results
    
    def test_auxiliary_channel_dropout(self,
                                        satellite_data: torch.Tensor,
                                        auxiliary_data: torch.Tensor,
                                        targets: torch.Tensor,
                                        channel_names: List[str],
                                        drop_combinations: List[List[int]]) -> Dict:
        """
        Test model when specific auxiliary channels are missing.
        
        Helps identify which auxiliary features are most critical.
        
        Args:
            satellite_data: Shape (B, C, H, W)
            auxiliary_data: Shape (B, T, F)
            targets: Shape (B, H, W)
            channel_names: Names of auxiliary channels
            drop_combinations: List of channel index combinations to drop
            
        Returns:
            Dictionary with metrics for each combination
        """
        results = {}
        
        # Baseline
        with torch.no_grad():
            baseline_pred, _ = self.model(
                satellite_data.to(self.device),
                auxiliary_data.to(self.device)
            )
        baseline_metrics = compute_metrics(
            baseline_pred.cpu().numpy(),
            targets.numpy()
        )
        results['baseline'] = baseline_metrics.to_dict()
        
        # Test each combination
        for drop_indices in drop_combinations:
            dropped_aux = auxiliary_data.clone()
            dropped_aux[:, :, drop_indices] = 0  # Zero out dropped channels
            
            with torch.no_grad():
                pred, _ = self.model(
                    satellite_data.to(self.device),
                    dropped_aux.to(self.device)
                )
                
            metrics = compute_metrics(pred.cpu().numpy(), targets.numpy())
            
            dropped_names = [channel_names[i] for i in drop_indices]
            key = f'drop_{"_".join(dropped_names)}'
            results[key] = metrics.to_dict()
            results[key]['dropped_channels'] = dropped_names
            results[key]['rmse_degradation'] = (
                metrics.rmse / baseline_metrics.rmse - 1
            ) * 100
            
        return results
    
    def test_noise_injection(self,
                              satellite_data: torch.Tensor,
                              auxiliary_data: torch.Tensor,
                              targets: torch.Tensor,
                              noise_levels: List[float] = [0.05, 0.1, 0.2]) -> Dict:
        """
        Test model robustness to Gaussian noise.
        
        Args:
            satellite_data: Shape (B, C, H, W)
            auxiliary_data: Shape (B, T, F)
            targets: Shape (B, H, W)
            noise_levels: Standard deviation of noise (relative to data std)
            
        Returns:
            Dictionary with metrics for each noise level
        """
        results = {}
        
        # Baseline
        with torch.no_grad():
            baseline_pred, _ = self.model(
                satellite_data.to(self.device),
                auxiliary_data.to(self.device)
            )
        baseline_metrics = compute_metrics(
            baseline_pred.cpu().numpy(),
            targets.numpy()
        )
        results['baseline'] = baseline_metrics.to_dict()
        
        sat_std = satellite_data.std().item()
        aux_std = auxiliary_data.std().item()
        
        for noise_level in noise_levels:
            # Add noise to satellite
            noisy_sat = satellite_data + torch.randn_like(satellite_data) * (sat_std * noise_level)
            
            # Add noise to auxiliary
            noisy_aux = auxiliary_data + torch.randn_like(auxiliary_data) * (aux_std * noise_level)
            
            with torch.no_grad():
                pred, _ = self.model(
                    noisy_sat.to(self.device),
                    noisy_aux.to(self.device)
                )
                
            metrics = compute_metrics(pred.cpu().numpy(), targets.numpy())
            results[f'noise_{int(noise_level*100)}pct'] = metrics.to_dict()
            results[f'noise_{int(noise_level*100)}pct']['rmse_degradation'] = (
                metrics.rmse / baseline_metrics.rmse - 1
            ) * 100
            
        return results
    
    def full_stress_test(self,
                          satellite_data: torch.Tensor,
                          auxiliary_data: torch.Tensor,
                          targets: torch.Tensor,
                          channel_names: Optional[List[str]] = None) -> Dict:
        """
        Run comprehensive stress testing.
        
        Args:
            satellite_data: Shape (B, C, H, W)
            auxiliary_data: Shape (B, T, F)
            targets: Shape (B, H, W)
            channel_names: Optional names for auxiliary channels
            
        Returns:
            Comprehensive robustness report
        """
        if channel_names is None:
            channel_names = [f'ch_{i}' for i in range(auxiliary_data.shape[-1])]
            
        results = {
            'satellite_dropout': self.test_satellite_dropout(
                satellite_data, auxiliary_data, targets
            ),
            'noise_injection': self.test_noise_injection(
                satellite_data, auxiliary_data, targets
            )
        }
        
        # Test dropping individual important channels
        if len(channel_names) >= 5:
            drop_combinations = [[0], [1], [0, 1], [0, 1, 2, 3, 4]]
            results['auxiliary_dropout'] = self.test_auxiliary_channel_dropout(
                satellite_data, auxiliary_data, targets,
                channel_names, drop_combinations
            )
            
        return results
    
    def create_report(self, results: Dict) -> str:
        """Create formatted robustness report."""
        lines = []
        lines.append("=" * 70)
        lines.append("ROBUSTNESS STRESS TEST REPORT")
        lines.append("=" * 70)
        
        for test_name, test_results in results.items():
            lines.append(f"\n{test_name.upper().replace('_', ' ')}:")
            lines.append("-" * 50)
            
            for condition, metrics in test_results.items():
                rmse = metrics.get('rmse', 0)
                degradation = metrics.get('rmse_degradation', 0)
                lines.append(f"  {condition}: RMSE={rmse:.6f} (Delta={degradation:+.1f}%)")
                
        lines.append("=" * 70)
        return "\n".join(lines)
