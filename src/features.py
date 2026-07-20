"""
Feature Engineering module for Fire Prediction (Model A) and Confidence Classification (Model B).
Generates temporal lags, rolling statistics, sin/cos seasonal encodings,
spatial Moore neighbourhood fire aggregations, and point-level features.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import convolve2d

try:
    from src.config import (
        PROCESSED_DAILY_GRID, PROCESSED_CLEANED_POINTS,
        LAG_DAYS, ROLLING_WINDOWS, DATA_DIR
    )
except ImportError:
    from config import (
        PROCESSED_DAILY_GRID, PROCESSED_CLEANED_POINTS,
        LAG_DAYS, ROLLING_WINDOWS, DATA_DIR
    )


def compute_spatial_neighbourhood_features(df_daily):
    """
    Compute Moore neighbourhood (3x3 spatial window) fire metrics across the grid
    for each calendar date using 2D spatial convolution.
    
    Parameters:
    -----------
    df_daily : pd.DataFrame
        Daily lattice dataframe containing 'date', 'lat_idx', 'lon_idx', 'fire_count', 'max_frp'.
        
    Returns:
    --------
    pd.DataFrame
        Updated dataframe with 'neighbor_fire_count' and 'neighbor_max_frp' on the same day.
        (Note: lag is applied subsequently to guarantee zero data leakage).
    """
    print("[Features] Computing spatial 3x3 neighbourhood features via 2D convolution...")
    
    # 3x3 Moore neighbourhood kernel with center excluded
    kernel = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float32)
    
    max_lat_idx = df_daily['lat_idx'].max() + 1
    max_lon_idx = df_daily['lon_idx'].max() + 1
    
    # Pre-allocate dictionary mapping (date, cell_id) -> neighbor_fire_count
    neighbor_records = []
    
    # Group by date and process 2D grid slices
    dates_grouped = df_daily.groupby('date')
    for date_val, group in dates_grouped:
        grid_fires = np.zeros((max_lat_idx, max_lon_idx), dtype=np.float32)
        grid_frp = np.zeros((max_lat_idx, max_lon_idx), dtype=np.float32)
        
        # Populate current day matrices
        grid_fires[group['lat_idx'].values, group['lon_idx'].values] = group['fire_count'].values
        grid_frp[group['lat_idx'].values, group['lon_idx'].values] = group['max_frp'].values
        
        # 2D convolve for neighbor fire sum
        conv_fires = convolve2d(grid_fires, kernel, mode='same', boundary='fill', fillvalue=0)
        
        # 2D local max for neighbor max FRP (custom dilation without center)
        padded_frp = np.pad(grid_frp, pad_width=1, mode='constant', constant_values=0)
        # Shift in 8 directions and compute maximum
        shifts = [
            padded_frp[:-2, :-2], padded_frp[:-2, 1:-1], padded_frp[:-2, 2:],
            padded_frp[1:-1, :-2],                       padded_frp[1:-1, 2:],
            padded_frp[2:, :-2],  padded_frp[2:, 1:-1],  padded_frp[2:, 2:]
        ]
        conv_max_frp = np.maximum.reduce(shifts)
        
        # Extract values for current group cells
        cell_ids = group['cell_id'].values
        lat_idxs = group['lat_idx'].values
        lon_idxs = group['lon_idx'].values
        
        n_fires = conv_fires[lat_idxs, lon_idxs]
        n_frp = conv_max_frp[lat_idxs, lon_idxs]
        
        sub_df = pd.DataFrame({
            'date': date_val,
            'cell_id': cell_ids,
            'spatial_neighbor_fire_count': n_fires,
            'spatial_neighbor_max_frp': n_frp
        })
        neighbor_records.append(sub_df)
        
    df_neighbors = pd.concat(neighbor_records, ignore_index=True)
    df_merged = df_daily.merge(df_neighbors, on=['date', 'cell_id'], how='left')
    return df_merged


def build_model_a_features(df_daily, lags=LAG_DAYS, windows=ROLLING_WINDOWS):
    """
    Construct all tabular features and target labels for Model A (Fire Risk Forecasting).
    
    Includes:
    - Target: next_day_fire (0 or 1), next_day_frp (float)
    - Lag features (1, 3, 7 days) of fire count, max FRP, brightness
    - Rolling means (3, 7 days) strictly shifted to avoid leakage
    - Sin/Cos seasonal cyclical features (day of year, month)
    - Spatial neighbour fire lag features (lag 1, lag 3)
    
    Parameters:
    -----------
    df_daily : pd.DataFrame
        Daily lattice dataset.
    lags : list of int
        Days to lag (default [1, 3, 7]).
    windows : list of int
        Rolling window sizes (default [3, 7]).
        
    Returns:
    --------
    pd.DataFrame
        Feature-engineered dataset ready for Model A training and evaluation.
    """
    print("[Features] Engineering temporal lags, rolling statistics, and targets for Model A...")
    
    # Ensure correct chronological order within each cell
    df = df_daily.sort_values(['cell_id', 'date']).reset_index(drop=True)
    
    # 1. Spatial Moore neighbourhood fire counts
    df = compute_spatial_neighbourhood_features(df)
    
    # Group by cell_id for time series operations
    grouped = df.groupby('cell_id')
    
    # 2. Target Variables (t+1 forecast)
    # Binary fire presence next day
    df['target_fire_next_day'] = (grouped['fire_count'].shift(-1) > 0).astype(float)
    # Continuous peak FRP next day
    df['target_frp_next_day'] = grouped['max_frp'].shift(-1).astype(float)
    
    # 3. Lag Features (t-k)
    for lag in lags:
        df[f'fire_count_lag_{lag}'] = grouped['fire_count'].shift(lag)
        df[f'max_frp_lag_{lag}'] = grouped['max_frp'].shift(lag)
        df[f'mean_brightness_lag_{lag}'] = grouped['mean_brightness'].shift(lag)
        df[f'mean_bright_t31_lag_{lag}'] = grouped['mean_bright_t31'].shift(lag)
        df[f'neighbor_fire_count_lag_{lag}'] = grouped['spatial_neighbor_fire_count'].shift(lag)
        df[f'neighbor_max_frp_lag_{lag}'] = grouped['spatial_neighbor_max_frp'].shift(lag)
        
    # 4. Rolling Statistics (strictly shift by 1 day first to prevent data leakage)
    for win in windows:
        # Fire count rolling mean & max
        shifted_fc = grouped['fire_count'].shift(1)
        df[f'fire_count_roll_mean_{win}d'] = shifted_fc.groupby(df['cell_id']).transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        df[f'fire_count_roll_max_{win}d'] = shifted_fc.groupby(df['cell_id']).transform(
            lambda x: x.rolling(win, min_periods=1).max()
        )
        
        # Max FRP rolling mean
        shifted_frp = grouped['max_frp'].shift(1)
        df[f'max_frp_roll_mean_{win}d'] = shifted_frp.groupby(df['cell_id']).transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        
        # Mean brightness rolling mean
        shifted_br = grouped['mean_brightness'].shift(1)
        df[f'mean_brightness_roll_mean_{win}d'] = shifted_br.groupby(df['cell_id']).transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        
    # 5. Sin/Cos Cyclical Seasonal Features
    doy = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2.0 * np.pi * doy / 365.25)
    df['cos_doy'] = np.cos(2.0 * np.pi * doy / 365.25)
    
    month = df['date'].dt.month
    df['sin_month'] = np.sin(2.0 * np.pi * month / 12.0)
    df['cos_month'] = np.cos(2.0 * np.pi * month / 12.0)
    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 6. Coordinate / Static Spatial Features
    # Normalized latitude and longitude (useful for tree and neural models)
    df['norm_cell_lat'] = (df['cell_lat'] - df['cell_lat'].min()) / (df['cell_lat'].max() - df['cell_lat'].min())
    df['norm_cell_lon'] = (df['cell_lon'] - df['cell_lon'].min()) / (df['cell_lon'].max() - df['cell_lon'].min())
    
    # Filter rows with complete lag history (drop initial max(lags) days)
    # and drop final day (where target t+1 is NaN)
    max_lag = max(lags)
    min_date = df['date'].min() + pd.Timedelta(days=max_lag)
    max_date = df['date'].max() - pd.Timedelta(days=1)
    
    clean_mask = (df['date'] >= min_date) & (df['date'] <= max_date)
    df_clean = df[clean_mask].copy().reset_index(drop=True)
    
    # Cast target to integer
    df_clean['target_fire_next_day'] = df_clean['target_fire_next_day'].astype(int)
    
    print(f"[Features] Model A features ready. Valid sample count: {len(df_clean):,}")
    print(f"[Features] Date range: {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}")
    print(f"[Features] Target distribution: {df_clean['target_fire_next_day'].value_counts().to_dict()} (Positive: {df_clean['target_fire_next_day'].mean()*100:.2f}%)")
    
    return df_clean


def build_model_b_features(df_points):
    """
    Construct point-level tabular features and target labels for Model B
    (Fire Confidence Multi-Class Classification: Low, Nominal, High).
    
    Parameters:
    -----------
    df_points : pd.DataFrame
        Cleaned point fire observations.
        
    Returns:
    --------
    tuple (pd.DataFrame, list of str, list of str)
        (DataFrame with features & target, full_feature_cols, no_brightness_feature_cols)
    """
    print("[Features] Engineering point-level features for Model B (Confidence Classification)...")
    df = df_points.copy()
    
    # Map confidence to numerical class labels
    # 'l' (low) -> 0, 'n' (nominal) -> 1, 'h' (high) -> 2
    conf_mapping = {'l': 0, 'n': 1, 'h': 2}
    df['target_confidence'] = df['confidence'].map(conf_mapping)
    
    # Time of day (fractional hour in UTC)
    df['hour'] = df['acq_datetime'].dt.hour + df['acq_datetime'].dt.minute / 60.0
    df['month'] = df['acq_datetime'].dt.month
    
    # Day/Night binary flag
    df['is_day'] = (df['daynight'] == 'D').astype(int)
    
    # Full feature set (including brightness & bright_t31)
    full_feature_cols = [
        'brightness', 'bright_t31', 'frp', 'scan', 'track',
        'is_day', 'latitude', 'longitude', 'hour', 'month'
    ]
    
    # Ablation feature set (brightness and bright_t31 removed)
    no_brightness_cols = [
        'frp', 'scan', 'track',
        'is_day', 'latitude', 'longitude', 'hour', 'month'
    ]
    
    print(f"[Features] Model B sample count: {len(df):,}")
    print(f"[Features] Class counts: {df['target_confidence'].value_counts().to_dict()} (0=Low, 1=Nominal, 2=High)")
    
    return df, full_feature_cols, no_brightness_cols


def run_feature_engineering():
    """Main pipeline execution for feature generation."""
    print("=" * 70)
    print("RUNNING STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    df_daily = pd.read_parquet(PROCESSED_DAILY_GRID)
    df_points = pd.read_parquet(PROCESSED_CLEANED_POINTS)
    
    df_model_a = build_model_a_features(df_daily)
    model_a_path = DATA_DIR / "processed" / "model_a_features.parquet"
    df_model_a.to_parquet(model_a_path, index=False)
    print(f"[Features] Saved Model A features dataset to {model_a_path}")
    
    df_model_b, full_b_cols, no_b_cols = build_model_b_features(df_points)
    model_b_path = DATA_DIR / "processed" / "model_b_features.parquet"
    df_model_b.to_parquet(model_b_path, index=False)
    print(f"[Features] Saved Model B features dataset to {model_b_path}")
    
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return df_model_a, df_model_b


if __name__ == "__main__":
    run_feature_engineering()
