# Cross-Modal Attention Framework for NO₂ Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready implementation of a **Cross-Modal Attention Framework** for predicting tropospheric NO₂ column density from Sentinel-5P TROPOMI satellite data fused with meteorological auxiliary features.

## Table of Contents

1. [Overview](#overview)
2. [Scientific Background](#scientific-background)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Detailed Usage](#detailed-usage)
7. [Project Structure](#project-structure)
8. [Data Pipeline](#data-pipeline)
9. [Model Architecture Details](#model-architecture-details)
10. [Training Regime](#training-regime)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Ablation Studies](#ablation-studies)
13. [Robustness Testing](#robustness-testing)
14. [Configuration](#configuration)
15. [Extending for Real Data](#extending-for-real-data)
16. [Troubleshooting](#troubleshooting)
17. [References](#references)

---

## Overview

This framework implements a state-of-the-art cross-modal attention mechanism that fuses:
- **Satellite observations**: Sentinel-5P TROPOMI Level-2 NO₂ tropospheric column measurements
- **Auxiliary features**: Meteorological data (wind, boundary layer height, temperature, etc.)

The key innovation is using **cross-modal attention** where spatial satellite tokens query temporal meteorological tokens, learning which weather conditions are most relevant for predicting NO₂ at each location.

### Key Features

- ✅ **Complete 15-step implementation** following the actionable plan
- ✅ **QA filtering** before any aggregation (avoid label leakage)
- ✅ **Area-weighted regridding** (no bilinear interpolation)
- ✅ **Robust normalization** with median/IQR (outlier resistant)
- ✅ **Multi-scale CNN encoder** matching atmospheric transport scales
- ✅ **Transformer-based auxiliary encoder** with temporal embeddings
- ✅ **Inspectable cross-modal attention** (not naive concatenation)
- ✅ **Warm-start training regime** (prevents attention collapse)
- ✅ **Comprehensive evaluation** by region, season, and pollution level
- ✅ **Robustness testing** with dropout and noise injection
- ✅ **Ablation studies** to validate architecture necessity

---

## Scientific Background

### NO₂ and Air Quality

Nitrogen dioxide (NO₂) is a key air pollutant and precursor to ground-level ozone. Tropospheric NO₂ is primarily produced by:
- Combustion processes (vehicles, power plants, industrial facilities)
- Natural sources (lightning, soil emissions)

### Sentinel-5P TROPOMI

The TROPOspheric Monitoring Instrument (TROPOMI) on Sentinel-5P provides:
- **Daily global coverage** (since October 2017)
- **High spatial resolution** (~5.5 km × 3.5 km at nadir)
- **Tropospheric NO₂ column density** (mol/m²)

### Why Cross-Modal Attention?

NO₂ concentrations are influenced by:
1. **Emissions** (urban sources, industrial activity)
2. **Meteorology** (wind disperses, BLH affects mixing)
3. **Chemistry** (photolysis, reaction with VOCs)

Cross-modal attention allows the model to learn:
- *Which meteorological conditions* affect predictions at *which locations*
- *Temporal relationships* between weather patterns and NO₂ transport
- *Interpretable attention weights* for scientific validation

---

## Architecture

```
                                  ┌─────────────────────────────────────┐
                                  │     Cross-Modal NO₂ Predictor      │
                                  └─────────────────────────────────────┘
                                                    │
                    ┌───────────────────────────────┴───────────────────────────────┐
                    │                                                               │
         ┌──────────▼──────────┐                                     ┌──────────────▼──────────────┐
         │   Satellite Input   │                                     │     Auxiliary Input         │
         │   NO₂ Field (H×W)   │                                     │   Meteorology (T×F)         │
         └──────────┬──────────┘                                     └──────────────┬──────────────┘
                    │                                                               │
         ┌──────────▼──────────┐                                     ┌──────────────▼──────────────┐
         │  Multi-Scale CNN    │                                     │   Transformer Encoder       │
         │  (3×3, 5×5, 7×7)    │                                     │   (4 layers, 8 heads)       │
         └──────────┬──────────┘                                     └──────────────┬──────────────┘
                    │                                                               │
                    │ Spatial Tokens (H×W, 256)                    Temporal Tokens (T, 256)
                    │                                                               │
                    └───────────────────────────┬───────────────────────────────────┘
                                                │
                                 ┌──────────────▼──────────────┐
                                 │    Cross-Modal Attention    │
                                 │  Q = Satellite, K/V = Aux   │
                                 │    (2 layers, 8 heads)      │
                                 └──────────────┬──────────────┘
                                                │
                                 ┌──────────────▼──────────────┐
                                 │    MLP Prediction Head      │
                                 │    (256 → 128 → 1)          │
                                 └──────────────┬──────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │   NO₂ Predictions     │
                                    │      (H × W)          │
                                    └───────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

### Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd "c:\Shine - Vijayan\2. M.TECH-AIML\CV_Assignment2"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## Quick Start

### Run Complete Pipeline (Synthetic Data Demo)

```bash
python main.py --mode full
```

This will:
1. Generate 1000 synthetic NO₂ samples with realistic patterns
2. Train the Cross-Modal Attention model (50 epochs)
3. Evaluate with comprehensive metrics
4. Run ablation studies and robustness testing
5. Generate visualizations to `outputs/`

### Quick Test (2 epochs)

```bash
python main.py --mode test
```

### Expected Output

```
======================================================================
  CROSS-MODAL ATTENTION FRAMEWORK FOR NO₂ PREDICTION
  Sentinel-5P TROPOMI + Meteorological Data Fusion
======================================================================

Step 1: Generating Synthetic Demonstration Data
--------------------------------------------------
  Generating 1000 samples...
  Grid size: 64x64
  Time steps: 24
  NO₂ field shape: (1000, 24, 64, 64)

Step 2: Feature Engineering
--------------------------------------------------
  Processed auxiliary shape: (1000, 24, 13)

Step 5: Training with Warm-Start Regime
--------------------------------------------------
Phase 1: Warmup (10 epochs)
  Epoch 5/10 - Train Loss: 0.043521 - Val Loss: 0.039842
  Epoch 10/10 - Train Loss: 0.028341 - Val Loss: 0.031256

Phase 2: Fine-tuning (40 epochs)
  Epoch 10/40 - Train Loss: 0.018234 - Val Loss: 0.021567
  ...

======================================================================
EVALUATION METRICS SUMMARY
======================================================================

GLOBAL METRICS:
----------------------------------------
  RMSE: 0.085432
  MAE:  0.062341
  R²:   0.8923
  Bias: -0.003421
  Spatial Correlation: 0.9456

REGIONAL BREAKDOWN:
----------------------------------------
Region          RMSE         MAE          R²        
urban           0.102345     0.078234     0.8654    
suburban        0.089123     0.065432     0.8821    
rural           0.072345     0.051234     0.9123    

======================================================================
  EXECUTION COMPLETE
======================================================================
```

---

## Detailed Usage

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --mode {full,test,eval}   Execution mode (default: full)
  --config PATH             Path to config file (default: config/config.yaml)
  --output PATH             Output directory (default: outputs)
```

### Python API Usage

```python
# Import modules
from src.models.fusion_model import CrossModalNO2Predictor
from src.training.trainer import Trainer, TrainerConfig
from src.evaluation.metrics import MetricsCalculator

# Create model
model = CrossModalNO2Predictor(
    satellite_in_channels=1,
    auxiliary_feature_dim=13,
    embed_dim=256,
    num_attention_heads=8,
    num_attention_layers=2
)

# Train
config = TrainerConfig(warmup_epochs=10, finetune_epochs=40)
trainer = Trainer(model, config)
history = trainer.train(train_loader, val_loader)

# Evaluate
predictions, attention = model(satellite_data, auxiliary_data)
metrics = MetricsCalculator().compute_all(predictions, targets)
```

---

## Project Structure

```
CV_Assignment2/
├── config/
│   └── config.yaml                 # Central configuration file
├── src/
│   ├── __init__.py
│   ├── data_pipeline/              # Step 1-6: Data ingestion & preprocessing
│   │   ├── __init__.py
│   │   ├── sentinel_ingestion.py   # TROPOMI L2 data loading, QA filtering
│   │   ├── spatial_regridding.py   # Area-weighted regridding
│   │   ├── temporal_alignment.py   # Satellite-auxiliary synchronization
│   │   ├── feature_engineering.py  # Wind encoding, BLH normalization
│   │   └── normalization.py        # Robust median/IQR scaling
│   ├── models/                     # Step 7-10: Model architecture
│   │   ├── __init__.py
│   │   ├── satellite_encoder.py    # Multi-scale CNN for spatial features
│   │   ├── auxiliary_encoder.py    # Transformer for temporal features
│   │   ├── cross_modal_attention.py # Core attention mechanism
│   │   ├── fusion_model.py         # Complete model + baselines
│   │   └── prediction_head.py      # Regression output layer
│   ├── training/                   # Step 11-12: Training regime
│   │   ├── __init__.py
│   │   ├── trainer.py              # Warm-start training loop
│   │   └── losses.py               # Huber, weighted MAE losses
│   ├── evaluation/                 # Step 13-15: Evaluation & validation
│   │   ├── __init__.py
│   │   ├── metrics.py              # RMSE, MAE, R² by region/season
│   │   ├── robustness_testing.py   # Stress testing with dropout
│   │   └── ablation.py             # Architecture validation
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py        # Plotting utilities
│       └── io_utils.py             # Config loading, result saving
├── outputs/                        # Generated outputs
│   ├── training_curves.png
│   ├── predictions_vs_actual.png
│   ├── spatial_comparison.png
│   ├── attention_maps.png
│   └── metrics.json
├── main.py                         # Main entry point
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## Data Pipeline

### Step 1: Sentinel-5P Data Ingestion

```python
from src.data_pipeline.sentinel_ingestion import SentinelDataLoader

loader = SentinelDataLoader(qa_threshold=0.75)
data = loader.load_tropomi_no2('S5P_OFFL_L2__NO2____*.nc')
filtered = loader.apply_filtering(data)
```

**Key Principle**: Always filter by QA value BEFORE any aggregation or interpolation.

### Step 2: Spatial Regridding

```python
from src.data_pipeline.spatial_regridding import AreaWeightedRegridder, create_target_grid

grid = create_target_grid({'lat_min': 35, 'lat_max': 55, 'lon_min': -15, 'lon_max': 25}, 
                          resolution=0.1)
regridder = AreaWeightedRegridder(grid)
gridded_no2, counts = regridder.regrid(no2_data, lats, lons)
```

**Key Principle**: Use area-weighted averaging, NOT bilinear interpolation.

### Step 3: Temporal Alignment

```python
from src.data_pipeline.temporal_alignment import TemporalAligner

aligner = TemporalAligner(window_hours=1.0, method='nearest')
aligned_aux = aligner.align_to_overpass(aux_data, aux_times, satellite_overpass_time)
```

**Key Principle**: Use nearest-neighbor or windowed aggregation, NOT linear interpolation.

### Step 4: Feature Engineering

```python
from src.data_pipeline.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.transform(aux_data)  # Returns (T, 13) array
```

Features:
- Wind: u_norm, v_norm, sin, cos, magnitude
- Atmospheric: BLH, temperature, humidity, pressure
- Temporal: day_sin, day_cos, hour_sin, hour_cos

### Step 5: Robust Normalization

```python
from src.data_pipeline.normalization import RobustNormalizer

normalizer = RobustNormalizer(clip_range=(-10, 10))
normalizer.fit(training_data)
normalized = normalizer.transform(data)
```

**Key Principle**: Use median/IQR (robust to outliers), NOT mean/std.

---

## Model Architecture Details

### Satellite Encoder (CNN)

- **Input**: NO₂ field (B, 1, H, W)
- **Multi-scale convolutions**: 3×3, 5×5, 7×7 kernels (captures 30-70km at 0.1°)
- **Progressive channels**: 32 → 64 → 128 → 256
- **Residual connections**: Prevent gradient degradation
- **Output**: Spatial tokens (B, H×W, 256)

### Auxiliary Encoder (Transformer)

- **Input**: Meteorological features (B, T, 13)
- **Learnable positional encoding**: For temporal structure
- **4 Transformer layers**: 8-head self-attention
- **Output**: Temporal tokens (B, T, 256)

### Cross-Modal Attention

- **Queries**: From satellite encoder (spatial)
- **Keys/Values**: From auxiliary encoder (temporal)
- **2 attention layers**: Each with 8 heads
- **Output**: Fused tokens (B, H×W, 256)

**Critical**: This is NOT concatenation. Attention weights are inspectable and validated.

### Prediction Head

- **MLP**: 256 → 128 → 1
- **Output**: NO₂ prediction per spatial token
- **Reshape**: (B, H×W, 1) → (B, H, W)

---

## Training Regime

### Two-Phase Warm-Start Strategy

1. **Phase 1 - Warmup** (10 epochs, LR=1e-3):
   - Train all parameters together
   - Higher learning rate for rapid initial learning
   
2. **Phase 2 - Fine-tuning** (40 epochs, LR=1e-4):
   - Reduced learning rate
   - Focus on attention and prediction head

**Why?** Joint cold-start training causes attention collapse (uniform weights).

### Loss Function

**Huber Loss** (default δ=1.0):
```
L = 0.5 * x²           if |x| ≤ δ
L = δ * (|x| - 0.5δ)   otherwise
```

- Smooth like MSE for small errors
- Robust like MAE for large errors (pollution spikes)

---

## Evaluation Metrics

### Global Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | √(mean((pred - target)²)) | Lower is better |
| MAE | mean(|pred - target|) | Lower is better |
| R² | 1 - SS_res/SS_tot | Higher is better (max=1) |
| Bias | mean(pred - target) | Close to 0 is best |
| Spatial Correlation | corr(pred, target) | Higher is better |

### Regional Breakdown

Metrics are computed separately for:
- **Urban**: City centers with high emissions
- **Suburban**: Residential areas
- **Rural**: Agricultural/natural areas
- **Industrial**: Manufacturing zones
- **Background**: Clean reference areas

### Seasonal Breakdown

- **Winter** (DJF): Higher NO₂ due to heating, stable atmosphere
- **Spring** (MAM): Transition period
- **Summer** (JJA): Lower NO₂ due to photochemistry
- **Autumn** (SON): Transition period

---

## Ablation Studies

### Validating Cross-Modal Attention

The ablation module tests architecture necessity:

1. **Attention Analysis**: Check if attention collapses to uniform
   - Entropy ratio < 0.9 = GOOD (selective attention)
   - Entropy ratio > 0.9 = BAD (uniform = not learning)

2. **Baseline Comparison**:
   - vs Concatenation: >5% RMSE improvement expected
   - vs Satellite-only: Auxiliary data should help

3. **Architecture Validation**:
   - If removing attention doesn't hurt → Architecture invalid
   - If attention is uniform → Need to fix training

---

## Robustness Testing

### Satellite Dropout Test

Simulates missing observations (cloud cover):
```
Drop 10% pixels → Expected <10% RMSE increase
Drop 30% pixels → Expected <30% RMSE increase
Drop 50% pixels → Model should still produce reasonable predictions
```

### Auxiliary Channel Dropout

Tests sensitivity to specific meteorological features:
```
Drop wind components → High degradation expected
Drop BLH → Medium degradation expected
```

### Noise Injection

Tests resilience to measurement noise:
```
5% noise → <5% RMSE increase
10% noise → <10% RMSE increase
```

---

## Configuration

### config/config.yaml

```yaml
# Data Configuration
data:
  grid_resolution: 0.1  # degrees
  qa_threshold: 0.75

# Feature Configuration  
features:
  satellite:
    in_channels: 1
    embed_dim: 256
  auxiliary:
    feature_dim: 13
    num_layers: 4
    num_heads: 8

# Model Configuration
model:
  cross_attention:
    num_heads: 8
    dropout: 0.1
  prediction_head:
    hidden_dim: 128

# Training Configuration
training:
  warmup_epochs: 10
  finetune_epochs: 40
  warmup_lr: 1e-3
  finetune_lr: 1e-4
  batch_size: 16
  loss: "huber"
  huber_delta: 1.0
```

---

## Extending for Real Data

### Step 1: Register for Data Access

1. **Copernicus Data Space**: https://dataspace.copernicus.eu/
2. **Create account** and obtain API credentials

### Step 2: Download Sentinel-5P Data

```python
from sentinelsat import SentinelAPI

api = SentinelAPI('username', 'password', 'https://apihub.copernicus.eu/apihub')

products = api.query(
    area='POLYGON((lon1 lat1, lon2 lat2, ...))',
    date=('20230101', '20231231'),
    platformname='Sentinel-5',
    producttype='L2__NO2___'
)

api.download_all(products)
```

### Step 3: Download ERA5 Meteorology

Use Copernicus Climate Data Store (CDS) API:
```python
import cdsapi

c = cdsapi.Client()
c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis',
    'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', 
                 'boundary_layer_height', '2m_temperature'],
    'year': '2023',
    'month': ['01', '02', ...],
    'day': [...],
    'time': ['00:00', '01:00', ...],
    'format': 'netcdf',
}, 'download.nc')
```

### Step 4: Load Real Data

```python
from src.data_pipeline.sentinel_ingestion import SentinelDataLoader

loader = SentinelDataLoader(qa_threshold=0.75)
data = loader.load_tropomi_no2('path/to/S5P_*.nc')
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch size in config.yaml
- Use `--mode test` for smaller grid

**2. Training Loss Not Decreasing**
- Check learning rate (try lower)
- Ensure data normalization is correct
- Verify QA filtering threshold

**3. Attention Collapse (uniform weights)**
- Increase warmup epochs
- Reduce fine-tuning learning rate
- Check auxiliary data quality

**4. Poor Regional Performance**
- Check if training data covers all regions
- Consider region-specific loss weighting

---

## References

### Key Papers

1. **TROPOMI NO₂ Algorithm**: Boersma et al., "Improving algorithms and uncertainty estimates for satellite NO₂ retrievals"

2. **Cross-Modal Attention**: Vaswani et al., "Attention Is All You Need" (2017)

3. **Multi-Modal Fusion**: Xu et al., "Cross-Modal Attention for Remote Sensing"

### Data Sources

- **Sentinel-5P TROPOMI**: https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p
- **ERA5 Reanalysis**: https://cds.climate.copernicus.eu/

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

This implementation follows the 15-step actionable plan for Cross-Modal Attention Framework for NO₂ Prediction, emphasizing:
- Rigorous data handling (no interpolation before masking)
- Interpretable attention mechanisms
- Comprehensive evaluation methodology
- Scientific validation through ablation studies

---

**Author**: Cross-Modal NO₂ Prediction Framework  
**Version**: 1.0.0  
**Last Updated**: February 2026
