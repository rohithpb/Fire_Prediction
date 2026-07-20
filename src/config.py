"""
Configuration file for Fire Prediction & Confidence Classification System.
Defines geographic boundaries, grid settings, temporal split parameters,
feature definitions, and file paths.
"""

from pathlib import Path
import os
import random
import numpy as np
import torch

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "DL_FIRE_SV-C2_820633"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for path in [DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Raw Data Paths
ARCHIVE_CSV = RAW_DATA_DIR / "fire_archive_SV-C2_820633.csv"
NRT_CSV = RAW_DATA_DIR / "fire_nrt_SV-C2_820633.csv"

# Processed Data Paths
PROCESSED_CLEANED_POINTS = PROCESSED_DATA_DIR / "cleaned_fire_points.parquet"
PROCESSED_DAILY_GRID = PROCESSED_DATA_DIR / "daily_grid_features.parquet"
METRICS_JSON = OUTPUTS_DIR / "metrics_summary.json"

# Reproducibility Seed
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Geographic Bounds for South India (Kerala region)
LAT_MIN = 8.1
LAT_MAX = 14.6
LON_MIN = 74.4
LON_MAX = 77.9

# Spatial Grid Cell Size in degrees (~27.5 km)
CELL_SIZE = 0.25

# Temporal Split Ratios (Chronological, strictly non-overlapping)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Lag and Rolling Window configurations
LAG_DAYS = [1, 3, 7]
ROLLING_WINDOWS = [3, 7]

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
