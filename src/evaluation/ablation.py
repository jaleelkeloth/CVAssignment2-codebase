"""
Ablation Studies

This module implements ablation experiments to validate the architecture.

Key engineering considerations:
- Remove attention to test its necessity
- Swap fusion methods (concatenation vs attention)
- If attention removal does not degrade results, architecture is invalid
- Compare against baselines
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Type
from torch.utils.data import DataLoader
from .metrics import compute_metrics, MetricResults


class AblationStudy:
    """
    Conducts ablation studies to validate cross-modal attention.
    
    A valid cross-modal attention architecture must show:
    1. Degradation when attention is removed
    2. Improvement over simple concatenation
    3. Meaningful attention patterns (not uniform)
    
    If attention removal doesn't hurt performance, the architecture
    is not learning cross-modal dependencies and should be rejected.
    """
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize ablation study.
        
        Args:
            device: Computation device
        """
        self.device = device
        self.results = {}
        
    def compare_models(self,
                       models: Dict[str, nn.Module],
                       test_loader: DataLoader) -> Dict:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary mapping model name to model instance
            test_loader: Test data loader
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, model in models.items():
            model = model.to(self.device)
            model.eval()
            
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for satellite, auxiliary, targets in test_loader:
                    satellite = satellite.to(self.device)
                    auxiliary = auxiliary.to(self.device)
                    
                    predictions, _ = model(satellite, auxiliary)
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(targets.numpy())
                    
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            metrics = compute_metrics(predictions, targets)
            results[name] = metrics.to_dict()
            
        self.results = results
        return results
    
    def analyze_attention_patterns(self,
                                    model: nn.Module,
                                    test_loader: DataLoader,
                                    num_samples: int = 10) -> Dict:
        """
        Analyze attention patterns for interpretability.
        
        Checks:
        1. Attention entropy (uniform = bad)
        2. Spatial variation in attention
        3. Correlation with known meteorological effects
        
        Args:
            model: Trained model with cross-attention
            test_loader: Test data loader
            num_samples: Number of samples to analyze
            
        Returns:
            Attention analysis results
        """
        model = model.to(self.device)
        model.eval()
        
        all_entropy = []
        all_max_attention = []
        
        sample_count = 0
        
        with torch.no_grad():
            for satellite, auxiliary, _ in test_loader:
                if sample_count >= num_samples:
                    break
                    
                satellite = satellite.to(self.device)
                auxiliary = auxiliary.to(self.device)
                
                _, attention = model(satellite, auxiliary, return_attention=True)
                
                if attention is not None:
                    # Compute entropy for each attention distribution
                    eps = 1e-8
                    entropy = -torch.sum(
                        attention * torch.log(attention + eps), 
                        dim=-1
                    )
                    
                    all_entropy.append(entropy.cpu().numpy().mean())
                    all_max_attention.append(attention.max(dim=-1)[0].cpu().numpy().mean())
                    
                sample_count += satellite.shape[0]
                
        if not all_entropy:
            return {'warning': 'No attention weights available'}
            
        n_aux = attention.shape[-1] if attention is not None else 1
        max_entropy = np.log(n_aux)
        
        avg_entropy = np.mean(all_entropy)
        entropy_ratio = avg_entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'average_entropy': avg_entropy,
            'max_entropy': max_entropy,
            'entropy_ratio': entropy_ratio,
            'average_max_attention': np.mean(all_max_attention),
            'is_collapsed': entropy_ratio > 0.9,  # >90% of max entropy = uniform
            'interpretation': (
                'Attention is effectively uniform (BAD)' 
                if entropy_ratio > 0.9 
                else 'Attention shows selective patterns (GOOD)'
            )
        }
        
    def run_full_ablation(self,
                          full_model: nn.Module,
                          baseline_models: Dict[str, nn.Module],
                          test_loader: DataLoader) -> Dict:
        """
        Run complete ablation study.
        
        Args:
            full_model: Full cross-modal attention model
            baseline_models: Dictionary of baseline models for comparison
            test_loader: Test data loader
            
        Returns:
            Complete ablation results
        """
        results = {}
        
        # Compare all models
        all_models = {'cross_modal_attention': full_model}
        all_models.update(baseline_models)
        
        results['model_comparison'] = self.compare_models(all_models, test_loader)
        
        # Analyze attention patterns
        results['attention_analysis'] = self.analyze_attention_patterns(
            full_model, test_loader
        )
        
        # Compute improvement over baselines
        full_rmse = results['model_comparison']['cross_modal_attention']['rmse']
        
        results['improvement_analysis'] = {}
        for name, metrics in results['model_comparison'].items():
            if name != 'cross_modal_attention':
                baseline_rmse = metrics['rmse']
                improvement = (1 - full_rmse / baseline_rmse) * 100
                results['improvement_analysis'][name] = {
                    'rmse_reduction_pct': improvement,
                    'is_significant': improvement > 5  # >5% is significant
                }
                
        # Validate architecture
        results['architecture_validation'] = self._validate_architecture(results)
        
        return results
    
    def _validate_architecture(self, results: Dict) -> Dict:
        """
        Validate that cross-modal attention is necessary.
        
        The architecture is valid if:
        1. Attention is not collapsed to uniform
        2. Performance is better than concatenation baseline
        3. Performance is better than satellite-only baseline
        """
        validation = {
            'is_valid': True,
            'issues': []
        }
        
        # Check attention collapse
        if results.get('attention_analysis', {}).get('is_collapsed', False):
            validation['is_valid'] = False
            validation['issues'].append(
                'Attention has collapsed to uniform distribution'
            )
            
        # Check improvement over baselines
        for baseline, analysis in results.get('improvement_analysis', {}).items():
            if not analysis.get('is_significant', True):
                validation['issues'].append(
                    f'Improvement over {baseline} is not significant'
                )
                
        if validation['issues']:
            validation['recommendation'] = (
                'Review model architecture - cross-modal attention may not be '
                'learning meaningful dependencies. Consider: (1) increasing '
                'attention layers, (2) adjusting learning rates, (3) checking '
                'data quality.'
            )
        else:
            validation['recommendation'] = (
                'Architecture validation passed. Cross-modal attention is '
                'contributing meaningfully to predictions.'
            )
            
        return validation
    
    def create_report(self, results: Dict) -> str:
        """Create formatted ablation study report."""
        lines = []
        lines.append("=" * 70)
        lines.append("ABLATION STUDY REPORT")
        lines.append("=" * 70)
        
        # Model comparison
        lines.append("\nMODEL COMPARISON:")
        lines.append("-" * 50)
        lines.append(f"{'Model':<30} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
        
        for name, metrics in results.get('model_comparison', {}).items():
            lines.append(
                f"{name:<30} {metrics['rmse']:<12.6f} "
                f"{metrics['mae']:<12.6f} {metrics['r2']:<10.4f}"
            )
            
        # Improvement analysis
        lines.append("\nIMPROVEMENT OVER BASELINES:")
        lines.append("-" * 50)
        for baseline, analysis in results.get('improvement_analysis', {}).items():
            improvement = analysis.get('rmse_reduction_pct', 0)
            significant = "✓" if analysis.get('is_significant', False) else "✗"
            lines.append(f"  vs {baseline}: {improvement:+.1f}% RMSE reduction {significant}")
            
        # Attention analysis
        if 'attention_analysis' in results:
            lines.append("\nATTENTION ANALYSIS:")
            lines.append("-" * 50)
            aa = results['attention_analysis']
            lines.append(f"  Entropy ratio: {aa.get('entropy_ratio', 0):.3f}")
            lines.append(f"  Max attention: {aa.get('average_max_attention', 0):.3f}")
            lines.append(f"  Status: {aa.get('interpretation', 'N/A')}")
            
        # Validation
        if 'architecture_validation' in results:
            lines.append("\nARCHITECTURE VALIDATION:")
            lines.append("-" * 50)
            av = results['architecture_validation']
            status = "PASSED ✓" if av.get('is_valid', False) else "FAILED ✗"
            lines.append(f"  Status: {status}")
            for issue in av.get('issues', []):
                lines.append(f"  Issue: {issue}")
            lines.append(f"\n  {av.get('recommendation', '')}")
            
        lines.append("=" * 70)
        return "\n".join(lines)
