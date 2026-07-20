# NASA FIRMS VIIRS Active Fire Prediction & Confidence Classification System
### Advanced Spatio-Temporal Forecasting & Radiometric Analysis for South India (Kerala Region)

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14%2Bcpu-ee4c2c.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-brightgreen.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-yellowgreen.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

---

## 1. Executive Summary

This project delivers an end-to-end, production-grade Machine Learning and Deep Learning system for active wildfire forecasting and radiometric confidence classification across South India (Kerala region, Lat 8.1°–14.6°N, Lon 74.4°–77.9°E).

The project develops and evaluates **two distinct machine learning systems**:
1. **MODEL A (Main) – 24-Hour Spatio-Temporal Fire Risk Forecasting:**
   A regional lattice model that predicts whether a $0.25^\circ \times 0.25^\circ$ cell (~27.5 km) will experience at least one active fire detection on the next day ($t+1$). Handles severe class imbalance (~5.8% positive rate) using synthesized zero-fire negatives, strict chronological splitting without data leakage, and benchmarks 5 architectures (Logistic Regression, Random Forest, XGBoost, LightGBM, and a PyTorch GRU recurrent sequence network).
2. **MODEL B (Secondary) – Satellite Detection Confidence Multi-Class Classification:**
   A point-level model classifying VIIRS detection confidence (`l`: Low, `n`: Nominal, `h`: High). Includes a critical satellite radiometric ablation study demonstrating that withholding mid-infrared brightness temperatures causes a **34.9% drop in Macro-F1**, confirming the physical principles of VIIRS detection algorithms.

---

## 2. Dataset Description

The data originates from NASA's Fire Information for Resource Management System (FIRMS), captured by the Visible Infrared Imaging Radiometer Suite (VIIRS) aboard the Suomi National Polar-orbiting Partnership (S-NPP) satellite (375-meter spatial resolution, I-band):

| Dataset File | Time Span | Records | Key Characteristics |
| :--- | :--- | :--- | :--- |
| `fire_archive_SV-C2_820633.csv` | 2026-01-01 to 2026-06-30 | 13,534 | Standard science-quality archive; includes `type` column |
| `fire_nrt_SV-C2_820633.csv` | 2026-07-02 to 2026-10-03 | 805 | Near Real-Time (NRT) rapid product; `version="2.0NRT"` |
| **Combined Cleaned Points** | **2026-01-01 to 2026-10-03** | **14,339** | **Deduplicated, synchronized datetime, bounded** |

### Raw Attributes
- `latitude`, `longitude`: Fire pixel centroid coordinates (WGS84).
- `brightness`: I-4 channel brightness temperature in Kelvin (3.74 µm mid-infrared).
- `bright_t31`: I-5 channel brightness temperature in Kelvin (11.45 µm thermal infrared).
- `scan`, `track`: Across-track and along-track pixel dimensions in kilometers.
- `acq_date`, `acq_time`: Acquisition date and UTC time (HHMM integer).
- `confidence`: Satellite detection confidence tier (`l` = Low, `n` = Nominal, `h` = High).
- `frp`: Fire Radiative Power in Megawatts (MW), measuring instantaneous radiant heat release.
- `daynight`: Orbit overpass timing (`D` = Daytime ~13:30 local, `N` = Nighttime ~01:30 local).
- `type`: Presumed source category (archive only; 0 = vegetation fire, 2 = other static, 3 = offshore).

---

## 3. Methodology & Technical Architecture

### 3.1 Spatio-Temporal Gridding & Negative Sampling (Model A)
Because active-fire satellite feeds only record fire detections (zero negative observations), models trained directly on point detections suffer from extreme sampling bias.
- **Geographic Grid:** The domain [8.1°–14.6°N, 74.4°–77.9°E] is discretized into cells of size $0.25^\circ \times 0.25^\circ$ (26 latitude bins $\times$ 14 longitude bins = **364 cells**).
- **Daily Aggregation:** For each cell and calendar day $t$, we compute: `fire_count`, `mean_frp`, `max_frp`, `mean_brightness`, `mean_bright_t31`, `day_night_ratio`, and `high_conf_share`.
- **Full Cartesian Product:** Cross-joining all 276 days against all 364 cells yields **100,464 cell-day observations**.
- **Negative Sample Synthesis:** Zero-fire cells are explicitly imputed with zeros (`fire_count=0`, `max_frp=0.0`, `has_fire=0`).
  - Active Fire Cell-Days: **5,907 (5.88%)**
  - Zero-Fire Negative Cell-Days: **94,557 (94.12%)**

### 3.2 Feature Engineering
To prevent temporal lookahead bias and data leakage, all features for forecasting day $t+1$ use information strictly available up to day $t$:
1. **Temporal Lags ($t-1, t-3, t-7$):**
   - Fire count lag: `fire_count_lag_1`, `fire_count_lag_3`, `fire_count_lag_7`
   - Peak FRP lag: `max_frp_lag_1`, `max_frp_lag_3`, `max_frp_lag_7`
   - Radiometric brightness lag: `mean_brightness_lag_1`, `mean_brightness_lag_7`
2. **Rolling Summary Windows (3-day and 7-day):**
   - Rolling mean and rolling maximum of fire count and FRP, strictly calculated using `shift(1).rolling(w)`.
3. **Spatial Moore Neighbourhoods ($3 \times 3$ Spatial Window):**
   - Implemented via fast 2D spatial convolution (`scipy.signal.convolve2d`) with an 8-neighbor Moore kernel.
   - Computes `neighbor_fire_count_lag_1` and `neighbor_max_frp_lag_1`, capturing fire spread from adjacent cells.
4. **Seasonal & Astronomical Encodings:**
   - Day-of-year cyclical transformations: $\sin(2\pi \cdot \text{DOY} / 365.25)$, $\cos(2\pi \cdot \text{DOY} / 365.25)$.
   - Month cyclical transformations: $\sin(2\pi \cdot \text{Month} / 12)$, $\cos(2\pi \cdot \text{Month} / 12)$.
   - Day of week and weekend indicator.

### 3.3 Leakage Prevention & Chronological Splitting
Random $k$-fold cross-validation or random train-test splitting causes catastrophic temporal data leakage in time-series spatial data (where yesterday's fire predicts today's fire in the same cell).
We enforce strict **chronological splitting** across calendar days:
- **Train Set (Earliest 70%):** Jan 08, 2026 – Jul 13, 2026 (68,068 samples, **7.64% positive rate**)
- **Validation Set (Next 15%):** Jul 14, 2026 – Aug 22, 2026 (14,560 samples, **1.38% positive rate**)
- **Test Set (Final 15%):** Aug 23, 2026 – Oct 02, 2026 (14,924 samples, **1.70% positive rate**)

> **Domain Insight:** The steep drop in positive rate from 7.64% (dry season) to 1.38%–1.70% reflects the arrival of the Southwest Monsoon in Kerala. This severe seasonal distribution shift makes the test benchmark an authentic real-world challenge.

---

## 4. Benchmark Results

### 4.1 Model A – 24-Hour Fire Risk Forecasting Comparison

All models were evaluated on the held-out chronological test set (14,924 samples, 254 positive fire events):

| Model Architecture | Test ROC-AUC | Test PR-AUC | Precision (0.50) | Recall (0.50) | F1 (0.50) | Optimal Thresh | Optimal F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch GRU (Deep Sequence)** | **0.8559** | **0.0926** | **0.1263** | **0.1850** | **0.1502** | **0.40** | **0.1683** |
| **Logistic Regression (Balanced)** | 0.8120 | 0.0556 | 0.0000 | 0.0000 | 0.0000 | 0.10 | 0.0967 |
| **Random Forest (Balanced Subsample)** | 0.7890 | 0.0618 | 0.1905 | 0.0157 | 0.0291 | 0.30 | 0.1089 |
| **LightGBM (scale_pos_weight)** | 0.7636 | 0.0528 | 0.0000 | 0.0000 | 0.0000 | 0.10 | 0.1146 |
| **XGBoost (scale_pos_weight)** | 0.7300 | 0.0444 | 0.0000 | 0.0000 | 0.0000 | 0.15 | 0.0973 |

*No-skill baseline PR-AUC (random guessing on test set) = **0.0170** (1.70%).*

#### Key Takeaways:
- **PyTorch GRU is the superior model**, achieving the highest discrimination (**ROC-AUC 0.8559**) and precision-recall trade-off (**PR-AUC 0.0926**, over **5.4× random chance**). Its 2-layer recurrent gating mechanism effectively models the multi-day temporal decay of fuel drying and ignition persistence.
- Due to severe class imbalance (1.7% positives), standard 0.50 thresholding suppresses recall on tree models; threshold calibration to the optimal operating point (0.10–0.40) boosts tree F1-scores to ~0.10–0.11, while GRU achieves **0.1683 optimal F1**.

---

### 4.2 Model B – Fire Detection Confidence Classification & Radiometric Ablation

Evaluating point detections on chronological test data (2,151 detections):

| Experiment Configuration | Macro-F1 | Weighted-F1 | Low F1 (`l`) | Nominal F1 (`n`) | High F1 (`h`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model B: Full Features (WITH Brightness)** | **0.9180** | **0.9451** | **0.7938** | **0.9671** | **0.9931** |
| **Model B: Ablation (WITHOUT Brightness)** | **0.5687** | **0.7398** | **0.3915** | **0.8039** | **0.5106** |
| **Performance Delta ($\Delta$)** | **-0.3493** | **-0.2053** | **-0.4023** | **-0.1632** | **-0.4825** |

#### Radiometric Interpretation:
VIIRS fire detection algorithms rely on the contextual temperature difference between the active flaming pixel (measured in the 3.74 µm I-4 band) and surrounding non-burning background pixels (measured in the 11.45 µm I-5 thermal band).
- When brightness temperature is removed, the model loses the physical combustion signal, resulting in a **34.9% drop in Macro-F1** and an almost complete failure to distinguish High-confidence events (F1 plummets from 0.993 to 0.510).
- This empirically proves that satellite detection confidence is fundamentally radiometric rather than purely spatial or temporal.

---

## 5. Visualizations Gallery

All figures are automatically generated and saved at publication resolution (300 DPI) in `outputs/`:

| Output File | Description | Preview / Insights |
| :--- | :--- | :--- |
| [`fire_locations_map.png`](outputs/fire_locations_map.png) | Active fire spatial scatter map across South India & Kerala | Highlights intense fire clusters along the Wayanad, Nilgiris, and Western Ghats forests. |
| [`monthly_fire_counts.png`](outputs/monthly_fire_counts.png) | Monthly fire count bar chart (Jan–Oct 2026) | Demonstrates strong seasonal peak in Feb–Apr followed by near-zero detections during monsoon. |
| [`hour_of_day_distribution.png`](outputs/hour_of_day_distribution.png) | UTC and Local Indian Standard Time (IST) diurnal profiles | Bimodal detection distribution corresponding to ~13:30 daytime and ~01:30 nighttime satellite overpasses. |
| [`frp_distribution.png`](outputs/frp_distribution.png) | Fire Radiative Power (MW) linear and Log10 distributions | Demonstrates heavy-tailed fire energy release (median ~2.5 MW, 95th percentile ~17.8 MW). |
| [`spatial_density_heatmap.png`](outputs/spatial_density_heatmap.png) | 2D Density Heatmap over $0.25^\circ \times 0.25^\circ$ grid cells | Visualizes fire hot-spots across the spatial lattice. |
| [`feature_importances.png`](outputs/feature_importances.png) | Random Forest & LightGBM feature importance rankings | Top predictors: 1-day lag fire count, 7-day rolling mean FRP, and spatial Moore neighbor fire counts. |
| [`roc_pr_curves.png`](outputs/roc_pr_curves.png) | Multi-model ROC and Precision-Recall Curves | Direct visual benchmark illustrating PyTorch GRU's superior AUC trajectory against baselines. |
| [`deep_model_training_curves.png`](outputs/deep_model_training_curves.png) | PyTorch GRU Training vs Validation loss & AUC curves | Confirms stable training convergence without overfitting over 12 epochs. |
| [`model_b_confusion_matrix.png`](outputs/model_b_confusion_matrix.png) | Side-by-side Confusion Matrix comparison (Ablation) | Displays misclassifications when brightness is removed. |

---

## 6. Project Directory Structure

```
Fire-Prediction/
├── DL_FIRE_SV-C2_820633/             # Raw NASA FIRMS VIIRS active fire data
│   ├── fire_archive_SV-C2_820633.csv # Archive dataset (Jan-Jun 2026, 13,534 rows)
│   ├── fire_nrt_SV-C2_820633.csv     # NRT dataset (Jul-Oct 2026, 805 rows)
│   └── Readme.txt                    # NASA FIRMS data notice
├── data/
│   └── processed/                    # Processed tabular datasets
│       ├── cleaned_fire_points.parquet
│       ├── daily_grid_features.parquet
│       ├── model_a_features.parquet
│       └── model_b_features.parquet
├── models/                           # Trained serialized models & preprocessors
│   ├── model_a_logistic_regression.joblib
│   ├── model_a_random_forest.joblib
│   ├── model_a_xgboost.joblib
│   ├── model_a_lightgbm.joblib
│   ├── model_a_pytorch_gru.pt        # Best Deep Learning Model weights
│   ├── model_b_with_brightness.joblib# Best Confidence Classifier
│   ├── model_b_no_brightness.joblib  # Ablation Classifier
│   ├── model_a_scaler.joblib         # Tabular feature standardizer
│   ├── gru_seq_scaler.joblib         # 3D Sequence standardizer
│   └── model_a_feature_names.json    # Explicit feature ordering
├── outputs/                          # High-resolution figures & metric exports
│   ├── fire_locations_map.png
│   ├── monthly_fire_counts.png
│   ├── hour_of_day_distribution.png
│   ├── frp_distribution.png
│   ├── spatial_density_heatmap.png
│   ├── feature_importances.png
│   ├── roc_pr_curves.png
│   ├── deep_model_training_curves.png
│   ├── model_b_confusion_matrix.png
│   ├── model_a_benchmark_table.csv
│   ├── model_b_benchmark_table.csv
│   └── metrics_summary.json
├── src/                              # Core clean modular implementation
│   ├── __init__.py
│   ├── config.py                     # Geographic bounds, seed, directory paths
│   ├── data_prep.py                  # Ingestion, deduplication, gridding, negative sampling
│   ├── features.py                   # 2D convolution neighbors, lags, rolling stats
│   ├── models_dl.py                  # PyTorch GRU recurrent architecture & sequence loaders
│   ├── train.py                      # Chronological splitting, class weighting, training
│   └── evaluate.py                   # Metrics calculation & visualization suite
├── data_prep.py                      # Top-level executable wrapper
├── features.py                       # Top-level executable wrapper
├── train.py                          # Top-level executable wrapper
├── evaluate.py                       # Top-level executable wrapper
├── predict.py                        # Operational CLI inference tool
├── run_pipeline.py                   # End-to-end one-command pipeline orchestrator
├── create_notebook.py                # Notebook generation utility
├── fire_prediction_pipeline.ipynb    # Fully executed Jupyter Notebook with charts & tables
├── requirements.txt                  # Pinned Python package dependencies
└── README.md                         # Project documentation and findings
```

---

## 7. How to Reproduce

### 7.1 Setup Environment

```bash
# Clone / navigate to project root
cd d:/Games/Fire-Prediction

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt
```

### 7.2 Run End-to-End Pipeline in One Command

```bash
python run_pipeline.py
```
This executes all stages sequentially:
1. `data_prep.py`: Harmonizes archive & NRT CSVs, forms $0.25^\circ$ daily lattice with negative samples.
2. `features.py`: Computes 2D spatial convolution Moore neighbor fire counts, temporal lags (1, 3, 7 days), rolling stats, and target variables.
3. `train.py`: Executes chronological 70/15/15 split, trains all Model A baselines + PyTorch GRU, trains Model B with and without brightness, and saves all models.
4. `evaluate.py`: Generates the benchmark comparison tables and renders all 9 visualization figures into `outputs/`.

### 7.3 Run Individual Modules Separately

```bash
python data_prep.py   # Ingestion & negative sampling
python features.py    # Feature engineering
python train.py       # Model training
python evaluate.py    # Tables & plots
```

### 7.4 Interactive Jupyter Notebook

Launch Jupyter to explore the interactive notebook with all executed charts, code cells, and narrative:
```bash
jupyter notebook fire_prediction_pipeline.ipynb
```

---

## 8. Operational Inference (`predict.py`)

The project includes an operational command-line interface (`predict.py`) allowing forest rangers, disaster management authorities, or researchers to query fire probability for any date and coordinate in South India.

### Usage:
```bash
python predict.py --date <YYYY-MM-DD> --lat <FLOAT> --lon <FLOAT> [--model gru|lgb|rf] [--json]
```

### Example: High-Risk Dry Season Query in Wayanad Forest Tract
```bash
python predict.py --date 2026-03-15 --lat 11.685 --lon 76.132 --model gru
```
**Output:**
```
======================================================================
          NASA FIRMS VIIRS ACTIVE FIRE RISK FORECAST
======================================================================
 Query Date        : 2026-03-15
 Forecast Target   : 2026-03-16 (Next-Day Risk)
 Target Coordinates: Lat 11.685°N, Lon 76.132°E
 Mapped Grid Cell  : cell_11.60_076.15 (Center: 11.725°N, 76.275°E)
----------------------------------------------------------------------
 FIRE PROBABILITY  : 34.71% (p = 0.3471)
 RISK LEVEL        : [YELLOW] MODERATE RISK
 EXPECTED PEAK FRP : 1.25 MW
 MODEL INFERENCE   : PyTorch GRU (Deep Recurrent Sequence Model)
----------------------------------------------------------------------
 Advisory          : Elevated thermal conditions. Recommend satellite surveillance and local ranger check.
======================================================================
```

### Example: Low-Risk Monsoon Query in Coastal Kochi
```bash
python predict.py --date 2026-08-15 --lat 9.931 --lon 76.267 --model gru
```
**Output:**
```
======================================================================
          NASA FIRMS VIIRS ACTIVE FIRE RISK FORECAST
======================================================================
 Query Date        : 2026-08-15
 Forecast Target   : 2026-08-16 (Next-Day Risk)
 Target Coordinates: Lat 9.931°N, Lon 76.267°E
 Mapped Grid Cell  : cell_09.85_076.15 (Center: 09.975°N, 76.275°E)
----------------------------------------------------------------------
 FIRE PROBABILITY  : 1.14% (p = 0.0114)
 RISK LEVEL        : [GREEN] LOW RISK
 EXPECTED PEAK FRP : 1.25 MW
 MODEL INFERENCE   : PyTorch GRU (Deep Recurrent Sequence Model)
----------------------------------------------------------------------
 Advisory          : Normal monitoring. Low likelihood of active open-burning or vegetation fires.
======================================================================
```

---

## 9. Limitations & Operational Considerations

1. **Cloud Cover & Monsoon Attenuation:**
   During the heavy Southwest Monsoon (June–September), dense cloud cover prevents thermal infrared sensors from detecting surface fires, potentially creating false negatives in satellite feeds.
2. **Satellite Revisit Temporal Gaps:**
   SNPP VIIRS provides approximately two daily observations (one daytime at ~13:30 local and one nighttime at ~01:30 local). Rapid burns initiated and extinguished between passes will not be captured.
3. **Absence of Surface Meteorology & Fuel Moisture:**
   The current models utilize purely endogenous satellite active fire histories, spatial propagation, and seasonal cycles. Integrating exogenous weather feeds (ERA5 reanalysis or IMD grid data: 2-meter temperature, relative humidity, wind velocity, precipitation, and soil moisture) will further improve PR-AUC during transition seasons.
4. **Topographic & Land Cover Constraints:**
   The Western Ghats feature complex micro-climates, elevation gradients, and varied fuel types (dry deciduous vs shola-grassland complexes). Adding Digital Elevation Model (DEM) slope/aspect and MODIS/Sentinel land cover masks would provide useful spatial priors.
