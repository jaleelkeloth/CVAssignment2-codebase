#!/usr/bin/env python
"""
Cross-Modal Attention Framework for NO2 Prediction

Main entry point for training and evaluating the Cross-Modal NO2 Predictor.

This script:
1. Generates synthetic demonstration data (or loads real data)
2. Trains the Cross-Modal Attention model with warm-start regime
3. Evaluates performance with comprehensive metrics
4. Runs ablation studies to validate architecture
5. Generates visualizations and reports

Usage:
    python main.py --mode full      # Run complete pipeline
    python main.py --mode test      # Quick test (2 epochs)
    python main.py --mode eval      # Evaluate pretrained model
"""

import argparse
import sys
import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline.sentinel_ingestion import SyntheticDataGenerator
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.normalization import RobustNormalizer

from src.models.fusion_model import (
    CrossModalNO2Predictor, 
    ConcatenationBaseline, 
    SatelliteOnlyBaseline
)

from src.training.trainer import Trainer, TrainerConfig, NO2Dataset
from src.training.losses import HuberLoss

from src.evaluation.metrics import MetricsCalculator, compute_metrics
from src.evaluation.robustness_testing import RobustnessTester
from src.evaluation.ablation import AblationStudy

from src.utils.visualization import Visualizer
from src.utils.io_utils import load_config, save_results, create_experiment_id


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("  CROSS-MODAL ATTENTION FRAMEWORK FOR NO2 PREDICTION")
    print("  Sentinel-5P TROPOMI + Meteorological Data Fusion")
    print("=" * 70)
    print()


def generate_demo_dataset(config: dict, verbose: bool = True):
    """
    Generate synthetic demonstration dataset.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, land_use_map)
    """
    if verbose:
        print("Step 1: Generating Synthetic Demonstration Data")
        print("-" * 50)
    
    synthetic_config = config.get('synthetic', {})
    num_samples = synthetic_config.get('num_samples', 1000)
    grid_height = synthetic_config.get('grid_height', 64)
    grid_width = synthetic_config.get('grid_width', 64)
    time_steps = synthetic_config.get('time_steps', 24)
    noise_level = synthetic_config.get('noise_level', 0.1)
    
    # Generate data
    generator = SyntheticDataGenerator(
        grid_height=grid_height,
        grid_width=grid_width,
        time_steps=time_steps,
        seed=42
    )
    
    if verbose:
        print(f"  Generating {num_samples} samples...")
        print(f"  Grid size: {grid_height}x{grid_width}")
        print(f"  Time steps: {time_steps}")
    
    no2_fields, aux_features_list, land_use = generator.generate_dataset(
        num_samples=num_samples,
        noise_level=noise_level
    )
    
    if verbose:
        print(f"  NO2 field shape: {no2_fields.shape}")
        print(f"  Auxiliary features: {len(aux_features_list)} samples")
        print(f"  Land use categories: {np.unique(land_use)}")
    
    # Process auxiliary features
    if verbose:
        print("\nStep 2: Feature Engineering")
        print("-" * 50)
    
    feature_engineer = FeatureEngineer()
    processed_aux = []
    for aux in aux_features_list:
        processed = feature_engineer.transform(aux)
        processed_aux.append(processed)
    processed_aux = np.stack(processed_aux, axis=0)  # (N, T, F)
    
    if verbose:
        print(f"  Processed auxiliary shape: {processed_aux.shape}")
        print(f"  Features: {feature_engineer.get_feature_names()}")
    
    # Normalize data
    if verbose:
        print("\nStep 3: Robust Normalization")
        print("-" * 50)
    
    sat_normalizer = RobustNormalizer()
    sat_normalizer.fit(no2_fields)
    no2_normalized = sat_normalizer.transform(no2_fields)
    
    if verbose:
        print(f"  Satellite - median: {sat_normalizer.stats.median:.6f}, IQR: {sat_normalizer.stats.iqr:.6f}")
    
    # Create PyTorch tensors
    # Use mean over time as satellite input, last time step as target
    satellite_input = no2_normalized.mean(axis=1, keepdims=True)  # (N, 1, H, W)
    satellite_input = torch.from_numpy(satellite_input).float()
    
    auxiliary_input = torch.from_numpy(processed_aux).float()  # (N, T, F)
    
    targets = torch.from_numpy(no2_normalized[:, -1, :, :]).float()  # (N, H, W)
    
    if verbose:
        print(f"  Satellite input: {satellite_input.shape}")
        print(f"  Auxiliary input: {auxiliary_input.shape}")
        print(f"  Targets: {targets.shape}")
    
    # Create dataset and split
    dataset = TensorDataset(satellite_input, auxiliary_input, targets)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = config.get('training', {}).get('batch_size', 16)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if verbose:
        print(f"\n  Train samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, land_use, sat_normalizer


def create_model(config: dict, device: str, verbose: bool = True):
    """Create the Cross-Modal NO2 Predictor model."""
    if verbose:
        print("\nStep 4: Creating Model Architecture")
        print("-" * 50)
    
    features_config = config.get('features', {})
    model_config = config.get('model', {})
    
    model = CrossModalNO2Predictor(
        satellite_in_channels=features_config.get('satellite', {}).get('in_channels', 1),
        auxiliary_feature_dim=13,  # From FeatureEngineer
        embed_dim=features_config.get('satellite', {}).get('embed_dim', 256),
        num_attention_heads=model_config.get('cross_attention', {}).get('num_heads', 8),
        num_attention_layers=2,
        satellite_hidden_dims=[32, 64, 128],
        auxiliary_num_layers=features_config.get('auxiliary', {}).get('num_layers', 4),
        prediction_hidden_dim=model_config.get('prediction_head', {}).get('hidden_dim', 128),
        dropout=model_config.get('satellite_encoder', {}).get('dropout', 0.1),
        use_lightweight=True  # For faster demo
    )
    
    model = model.to(device)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {device}")
    
    return model


def train_model(model, train_loader, val_loader, config, verbose=True):
    """Train the model with warm-start regime."""
    if verbose:
        print("\nStep 5: Training with Warm-Start Regime")
        print("-" * 50)
    
    training_config = config.get('training', {})
    
    trainer_config = TrainerConfig(
        warmup_epochs=training_config.get('warmup_epochs', 10),
        finetune_epochs=training_config.get('finetune_epochs', 40),
        warmup_lr=training_config.get('warmup_lr', 1e-3),
        finetune_lr=training_config.get('finetune_lr', 1e-4),
        weight_decay=training_config.get('weight_decay', 1e-5),
        huber_delta=training_config.get('huber_delta', 1.0),
        batch_size=training_config.get('batch_size', 16),
        save_checkpoints=False,  # Disable for demo
        log_freq=5
    )
    
    trainer = Trainer(model, trainer_config)
    history = trainer.train(train_loader, val_loader, verbose=verbose)
    
    return history, trainer


def evaluate_model(model, test_loader, land_use, device, verbose=True):
    """Evaluate model with comprehensive metrics."""
    if verbose:
        print("\nStep 6: Comprehensive Evaluation")
        print("-" * 50)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for satellite, auxiliary, targets in test_loader:
            satellite = satellite.to(device)
            auxiliary = auxiliary.to(device)
            
            predictions, _ = model(satellite, auxiliary)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Create metrics calculator with region mask
    # Broadcast land_use to match predictions shape
    region_mask = np.broadcast_to(land_use, predictions.shape)
    
    metrics_calculator = MetricsCalculator(
        region_mask=land_use,
        region_labels={0: 'background', 1: 'rural', 2: 'suburban', 3: 'urban', 4: 'industrial'}
    )
    
    results = metrics_calculator.compute_all(predictions, targets)
    
    if verbose:
        print(metrics_calculator.create_comparison_table(results))
    
    return results, predictions, targets


def run_ablation_study(model, test_loader, device, verbose=True):
    """Run ablation studies to validate architecture."""
    if verbose:
        print("\nStep 7: Ablation Studies")
        print("-" * 50)
    
    # Create baseline models
    baselines = {
        'concatenation': ConcatenationBaseline(
            satellite_in_channels=1,
            auxiliary_feature_dim=13,
            embed_dim=256
        ).to(device),
        'satellite_only': SatelliteOnlyBaseline(
            satellite_in_channels=1,
            embed_dim=256
        ).to(device)
    }
    
    # Note: In a real scenario, you would train these baselines
    # For demo, we just compare architectures
    
    ablation = AblationStudy(device=device)
    
    # Compare with full model
    all_models = {'cross_modal_attention': model}
    all_models.update(baselines)
    
    # This comparison is somewhat limited since baselines aren't trained
    ablation_results = {
        'attention_analysis': ablation.analyze_attention_patterns(model, test_loader)
    }
    
    if verbose:
        print("\n  Attention Analysis:")
        aa = ablation_results['attention_analysis']
        print(f"    Entropy ratio: {aa.get('entropy_ratio', 'N/A')}")
        print(f"    Status: {aa.get('interpretation', 'N/A')}")
    
    return ablation_results


def run_stress_testing(model, test_loader, device, verbose=True):
    """Run robustness stress tests."""
    if verbose:
        print("\nStep 8: Robustness Stress Testing")
        print("-" * 50)
    
    # Get a small batch for testing
    satellite, auxiliary, targets = next(iter(test_loader))
    
    tester = RobustnessTester(model, device=device)
    
    stress_results = tester.full_stress_test(
        satellite, auxiliary, targets,
        channel_names=FeatureEngineer.get_feature_names()
    )
    
    if verbose:
        print(tester.create_report(stress_results))
    
    return stress_results


def generate_visualizations(history, predictions, targets, attention, output_dir, verbose=True):
    """Generate all visualizations."""
    if verbose:
        print("\nStep 9: Generating Visualizations")
        print("-" * 50)
    
    viz = Visualizer(output_dir)
    
    # Training curves
    if history:
        path = viz.plot_training_curves(history)
        if verbose:
            print(f"  Training curves: {path}")
    
    # Predictions vs actual
    path = viz.plot_predictions_vs_actual(predictions, targets)
    if verbose:
        print(f"  Predictions scatter: {path}")
    
    # Spatial comparison (first sample)
    path = viz.plot_spatial_comparison(predictions[0], targets[0])
    if verbose:
        print(f"  Spatial comparison: {path}")
    
    # Attention maps
    if attention is not None:
        H, W = predictions.shape[1], predictions.shape[2]
        path = viz.plot_attention_map(
            attention[0].cpu().numpy() if torch.is_tensor(attention) else attention[0],
            spatial_shape=(H, W)
        )
        if verbose:
            print(f"  Attention maps: {path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Cross-Modal Attention Framework for NO2 Prediction'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='full',
        choices=['full', 'test', 'eval'],
        help='Execution mode: full (complete pipeline), test (quick test), eval (evaluate only)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Configuration loaded from: {config_path}")
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {}
    
    # Adjust for test mode
    if args.mode == 'test':
        print("\n[TEST MODE] Using reduced configuration for quick testing\n")
        config['synthetic'] = {
            'num_samples': 100,
            'grid_height': 32,
            'grid_width': 32,
            'time_steps': 12
        }
        config['training'] = {
            'warmup_epochs': 2,
            'finetune_epochs': 3,
            'batch_size': 8
        }
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate demonstration data
    train_loader, val_loader, test_loader, land_use, normalizer = generate_demo_dataset(config)
    
    # Create model
    model = create_model(config, device)
    
    # Train model
    history, trainer = train_model(model, train_loader, val_loader, config)
    
    # Get attention weights for visualization
    satellite, auxiliary, targets = next(iter(test_loader))
    with torch.no_grad():
        _, attention = model(satellite.to(device), auxiliary.to(device))
    
    # Evaluate
    eval_results, predictions, targets_np = evaluate_model(
        model, test_loader, land_use, device
    )
    
    # Ablation study
    ablation_results = run_ablation_study(model, test_loader, device)
    
    # Robustness testing
    stress_results = run_stress_testing(model, test_loader, device)
    
    # Generate visualizations
    generate_visualizations(
        history, predictions, targets_np, attention, args.output
    )
    
    # Save all results
    all_results = {
        'experiment_id': create_experiment_id(),
        'config': config,
        'evaluation': eval_results,
        'ablation': ablation_results,
        'robustness': stress_results,
        'training_history': {
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else None,
            'total_epochs': len(history['train_loss'])
        }
    }
    
    results_path = save_results(all_results, args.output, 'metrics.json')
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("  EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\n  Output directory: {output_dir.absolute()}")
    print("  Generated files:")
    for f in output_dir.glob("*"):
        print(f"    - {f.name}")
    print()


if __name__ == '__main__':
    main()
