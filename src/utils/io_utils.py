"""
I/O Utilities

Functions for configuration loading and result saving.
"""

import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict, 
                 output_dir: str,
                 filename: str = 'results.json') -> str:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    results_converted = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2)
        
    return str(output_path)


def create_experiment_id() -> str:
    """Create unique experiment ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint."""
    import torch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
