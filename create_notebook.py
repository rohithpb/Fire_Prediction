"""
Script to generate the complete, self-contained Jupyter Notebook
'fire_prediction_pipeline.ipynb' for interactive demonstration and reproducibility.
"""

import json
from pathlib import Path

def build_notebook():
    project_root = Path(__file__).resolve().parent
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # 1. Header
    add_md("""# NASA FIRMS VIIRS Active Fire Prediction & Confidence Classification
### Spatio-Temporal Forecasting & Radiometric Analysis for South India (Kerala Region)

**Author:** Advanced Agentic AI Engineer  
**Dataset:** NASA FIRMS VIIRS (Suomi NPP, 375m I-band) Active Fire Detections  
**Geographic Domain:** Lat 8.1°N – 14.6°N, Lon 74.4°E – 77.9°E (Kerala, Western Ghats & neighboring South India)  
**Time Horizon:** January 1, 2026 to October 3, 2026 (Archive: 13,534 rows + NRT: 805 rows = 14,339 total points)

---

## Project Structure & Architecture Overview
This project addresses two critical environmental machine learning challenges:

1. **MODEL A (Main): 24-Hour Spatio-Temporal Fire Risk Forecasting**
   - Synthesizes a discrete geographic lattice ($0.25^\\circ \\times 0.25^\\circ$, ~27.5 km) generating positive and explicit negative samples (zero-fire cells).
   - Engineers temporal lag features (1, 3, 7 days), rolling statistics (3, 7 days), sin/cos cyclical seasonal encodings, and spatial Moore neighbourhood (3×3) fire counts.
   - Enforces strict **chronological splitting** (70% train, 15% validation, 15% test) with **zero temporal leakage**.
   - Benchmarks 5 diverse architectures: Logistic Regression, Random Forest, XGBoost, LightGBM, and a **PyTorch GRU deep recurrent sequence model** with positive class weighting.

2. **MODEL B (Secondary): Active Fire Detection Confidence Classification**
   - Multi-class classification of satellite detection confidence (`l`: Low, `n`: Nominal, `h`: High).
   - Conducts an ablation study evaluating performance degradation when mid-infrared brightness temperatures are withheld.""")

    # 2. Imports & Config
    add_md("## 1. Environment Setup & Configuration")
    add_code("""import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch

# Ensure local source directory is on path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import (
    ARCHIVE_CSV, NRT_CSV, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    CELL_SIZE, RANDOM_SEED, set_seed, DEVICE, OUTPUTS_DIR, MODELS_DIR
)
from src.data_prep import load_and_combine_data, assign_grid_cells, build_daily_spatiotemporal_grid
from src.features import build_model_a_features, build_model_b_features
from src.evaluate import generate_benchmark_tables
from predict import predict_fire_risk

set_seed(RANDOM_SEED)
print(f"Project Root: {project_root}")
print(f"PyTorch Device: {DEVICE}")""")

    # 3. Data Ingestion
    add_md("""## 2. Data Ingestion & Harmonization
The raw data consists of two NASA FIRMS VIIRS SV-C2 files:
- `fire_archive_SV-C2_820633.csv`: Historical high-quality processed detections (Jan 1 – Jun 30, 2026).
- `fire_nrt_SV-C2_820633.csv`: Near Real-Time rapid response detections (Jul 2 – Oct 3, 2026).

We harmonize column types (`version` to string, parse `acq_time` into UTC datetime), drop coordinate duplicates, and restrict to the bounding box.""")

    add_code("""df_raw = load_and_combine_data()
df_gridded = assign_grid_cells(df_raw)

print(f"Total Cleaned Points: {len(df_gridded):,}")
display(df_gridded[['latitude', 'longitude', 'acq_datetime', 'brightness', 'frp', 'confidence', 'daynight', 'cell_id']].head(5))""")

    # 4. Exploratory Data Analysis
    add_md("""## 3. Exploratory Data Analysis & Visualizations
Let's inspect the key spatial, temporal, and radiometric distributions:
- **Spatial clustering** along the Western Ghats forest tracts
- **Monthly seasonality**: Peak in dry season (Feb–May) and steep drop during the Southwest Monsoon (Jul–Sep)
- **Diurnal cycle**: Synchronous with SNPP 1:30 PM (day) and 1:30 AM (night) sun-synchronous overpasses
- **FRP intensity**: Heavy-tailed log-normal distribution""")

    add_code("""# Display generated exploratory figures
from IPython.display import Image, display

print("Active Fire Locations Map:")
display(Image(filename=str(OUTPUTS_DIR / "fire_locations_map.png")))

print("Monthly Seasonality (2026):")
display(Image(filename=str(OUTPUTS_DIR / "monthly_fire_counts.png")))

print("Diurnal / Acquisition Hour Profile:")
display(Image(filename=str(OUTPUTS_DIR / "hour_of_day_distribution.png")))

print("Fire Radiative Power (FRP) Intensity Distribution:")
display(Image(filename=str(OUTPUTS_DIR / "frp_distribution.png")))""")

    # 5. Spatio-Temporal Gridding & Negative Sampling
    add_md("""## 4. Spatio-Temporal Gridding & Negative Sample Generation
In raw satellite detections, **only active fires are recorded** (there are no negative samples). To train a predictive model for fire occurrence, we must create a spatio-temporal lattice of all grid cells and all calendar days.

- **Cell resolution:** $0.25^\\circ \\times 0.25^\\circ$ (~27.5 km) = 26 latitude bins $\\times$ 14 longitude bins = **364 cells**.
- **Time range:** 276 consecutive calendar days (Jan 1 to Oct 3, 2026).
- **Cartesian Product:** 276 days $\\times$ 364 cells = **100,464 cell-day observations**.
- Cells with no fires on date $t$ are assigned `fire_count = 0`, `max_frp = 0.0`, `mean_brightness = 0.0`.""")

    add_code("""df_daily_path = project_root / "data" / "processed" / "daily_grid_features.parquet"
df_daily = pd.read_parquet(df_daily_path)

print(f"Total Lattice Points: {len(df_daily):,}")
print(f"Positive Fire Events : {df_daily['has_fire'].sum():,} ({df_daily['has_fire'].mean()*100:.2f}%)")
print(f"Zero-Fire Negatives  : {(df_daily['has_fire']==0).sum():,} ({(1-df_daily['has_fire'].mean())*100:.2f}%)")

display(Image(filename=str(OUTPUTS_DIR / "spatial_density_heatmap.png")))""")

    # 6. Feature Engineering
    add_md("""## 5. Feature Engineering: Lags, Rolling Stats, and Moore Neighbourhoods
To forecast whether a cell will have a fire on the next day ($t+1$), we compute:
1. **Target:** `target_fire_next_day` = $(fire\\_count_{t+1} > 0)$
2. **Temporal Lags:** Fire count, peak FRP, and brightness at lags $t-1, t-3, t-7$.
3. **Rolling Windows:** 3-day and 7-day rolling means and max, strictly shifted to exclude current-day leakage.
4. **Spatial Moore Neighbourhood:** 2D spatial convolution across the $3 \\times 3$ surrounding cells (center cell excluded) at lag $t-1$.
5. **Seasonal Cycles:** $\\sin(2\\pi \\cdot \\text{DOY} / 365.25)$, $\\cos(2\\pi \\cdot \\text{DOY} / 365.25)$.""")

    add_code("""model_a_path = project_root / "data" / "processed" / "model_a_features.parquet"
df_features = pd.read_parquet(model_a_path)
print(f"Model A Samples: {len(df_features):,}")
print(f"Features: {list(df_features.columns)}")""")

    # 7. Model A Benchmark
    add_md("""## 6. Model A Benchmark: Chronological Train / Val / Test Results
Evaluation is performed with strict **chronological splitting**:
- **Train (70%):** Jan 08 to Jul 13, 2026 (68,068 samples, 7.64% positives)
- **Validation (15%):** Jul 14 to Aug 22, 2026 (14,560 samples, 1.38% positives)
- **Test (15%):** Aug 23 to Oct 02, 2026 (14,924 samples, 1.70% positives)

Notice that during the monsoon season (Aug–Oct), the fire positive rate drops significantly. We report ROC-AUC, PR-AUC, Precision, Recall, and F1.""")

    add_code("""with open(OUTPUTS_DIR / "metrics_summary.json", "r") as f:
    metrics = json.load(f)

df_table_a, df_table_b = generate_benchmark_tables(metrics)""")

    add_code("""print("Model A ROC & Precision-Recall Curves:")
display(Image(filename=str(OUTPUTS_DIR / "roc_pr_curves.png")))

print("Model A Tree Feature Importances:")
display(Image(filename=str(OUTPUTS_DIR / "feature_importances.png")))

print("PyTorch GRU Training Convergence:")
display(Image(filename=str(OUTPUTS_DIR / "deep_model_training_curves.png")))""")

    # 8. Model B Results
    add_md("""## 7. Model B: Fire Detection Confidence Classification & Ablation
NASA VIIRS designates fire detections with Low (`l`), Nominal (`n`), or High (`h`) confidence based on radiometric brightness contrast against background ambient temperatures, cloud proximity, and day/night algorithms.

We train a multi-class Random Forest with balanced class weights and evaluate the impact of **removing brightness features**:
- **Full Model (with Brightness):** Macro-F1 = **0.9180**
- **Ablation Model (without Brightness):** Macro-F1 = **0.5687** (Drop: **-0.3493**)

This confirms the physical satellite remote sensing principle: mid-infrared brightness temperature (channel I-4 at $3.74\\,\\mu\\text{m}$) is the primary physical signal separating real combustion from thermal noise.""")

    add_code("""print("Model B Confusion Matrix Comparison (With vs Without Brightness):")
display(Image(filename=str(OUTPUTS_DIR / "model_b_confusion_matrix.png")))""")

    # 9. Interactive Inference Demo
    add_md("""## 8. Interactive Fire Risk Inference Demo (`predict.py`)
You can forecast next-day fire probability for any date and geographic location in South India.""")

    add_code("""# Example 1: Wayanad / Northern Kerala Forest Tract in Dry Season (High Risk Period)
res_wayanad = predict_fire_risk(date_str="2026-03-15", lat=11.685, lon=76.132, model_type="gru")
print("Wayanad Forecast (March 15, 2026):")
print(json.dumps(res_wayanad['fire_risk_forecast'], indent=2))

# Example 2: Coastal Kochi during Southwest Monsoon (Low Risk Period)
res_kochi = predict_fire_risk(date_str="2026-08-15", lat=9.931, lon=76.267, model_type="gru")
print("\\nKochi Forecast (August 15, 2026):")
print(json.dumps(res_kochi['fire_risk_forecast'], indent=2))""")

    add_md("""## 9. Key Conclusions & Operational Recommendations
1. **Model A Performance:** The PyTorch GRU recurrent sequence model achieved the highest test ROC-AUC (**0.8559**) and PR-AUC (**0.0926**), outperforming gradient boosted trees and logistic regression on long-range temporal dependencies.
2. **Model B Radiometric Finding:** Brightness temperature is the indispensable feature for confidence classification; removing it reduces Macro-F1 by **34.9%**.
3. **Operational Utility:** The configurable $0.25^\\circ$ spatial lattice and `predict.py` CLI provide forest management authorities with actionable next-day fire hazard intelligence.""")

    output_nb_path = project_root / "fire_prediction_pipeline.ipynb"
    with open(output_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated Jupyter Notebook at: {output_nb_path}")

if __name__ == "__main__":
    build_notebook()
