"""
Data preparation and spatio-temporal gridding module.
Handles loading, merging, datetime parsing, geographic gridding, daily aggregation,
and negative sample synthesis for active-fire observation data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from src.config import (
        ARCHIVE_CSV, NRT_CSV, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        CELL_SIZE, PROCESSED_CLEANED_POINTS, PROCESSED_DAILY_GRID
    )
except ImportError:
    from config import (
        ARCHIVE_CSV, NRT_CSV, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        CELL_SIZE, PROCESSED_CLEANED_POINTS, PROCESSED_DAILY_GRID
    )


def load_and_combine_data(archive_path=ARCHIVE_CSV, nrt_path=NRT_CSV):
    """
    Load archive and NRT fire detection CSVs, harmonize columns,
    parse timestamps into datetime, and deduplicate records.
    
    Parameters:
    -----------
    archive_path : Path or str
        Path to archive CSV (historical high-quality processed data)
    nrt_path : Path or str
        Path to NRT CSV (near real-time data, lacks 'type' column)
        
    Returns:
    --------
    pd.DataFrame
        Cleaned, deduplicated fire point detections dataframe.
    """
    print(f"[Data Prep] Loading archive data from {archive_path}...")
    df_archive = pd.read_csv(archive_path)
    print(f"[Data Prep] Loaded {len(df_archive):,} archive rows.")

    print(f"[Data Prep] Loading NRT data from {nrt_path}...")
    df_nrt = pd.read_csv(nrt_path)
    print(f"[Data Prep] Loaded {len(df_nrt):,} NRT rows.")

    # Combine datasets
    df = pd.concat([df_archive, df_nrt], ignore_index=True)
    initial_count = len(df)
    print(f"[Data Prep] Total combined records: {initial_count:,}")

    # Parse acq_date and acq_time (HHMM integer) into unified datetime
    # e.g., 839 -> '0839' -> 08:39 UTC
    time_str = df['acq_time'].astype(str).str.zfill(4)
    df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + time_str, format='%Y-%m-%d %H%M')
    df['date'] = pd.to_datetime(df['acq_date'])

    # Standardize string and mixed-type fields
    # Version in archive is '2' (int) and in NRT is '2.0NRT' (str), so standardize to str
    df['version'] = df['version'].astype(str)
    df['type'] = pd.to_numeric(df['type'], errors='coerce')
    df['confidence'] = df['confidence'].astype(str).str.strip().str.lower()
    df['daynight'] = df['daynight'].astype(str).str.strip().str.upper()

    # Filter bounding box (defensive guard)
    mask_bounds = (
        (df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
        (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
    )
    df = df[mask_bounds].copy()
    print(f"[Data Prep] Records within bounds [{LAT_MIN}, {LAT_MAX}] lat, [{LON_MIN}, {LON_MAX}] lon: {len(df):,}")

    # Drop duplicate detections if exact location and time match
    dup_cols = ['latitude', 'longitude', 'acq_datetime']
    dups_count = df.duplicated(subset=dup_cols).sum()
    if dups_count > 0:
        df = df.drop_duplicates(subset=dup_cols).reset_index(drop=True)
        print(f"[Data Prep] Removed {dups_count} duplicate detections.")
    else:
        print("[Data Prep] No exact coordinate-time duplicates found.")

    return df


def assign_grid_cells(df, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX, cell_size=CELL_SIZE):
    """
    Assign each fire detection coordinate to a discrete spatial grid cell.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Point fire detections with 'latitude' and 'longitude'.
    cell_size : float
        Grid cell resolution in degrees (default: 0.25).
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added cell coordinates and cell identifiers.
    """
    df = df.copy()
    
    # Calculate cell lower bound and indices
    df['lat_idx'] = np.floor((df['latitude'] - lat_min) / cell_size).astype(int)
    df['lon_idx'] = np.floor((df['longitude'] - lon_min) / cell_size).astype(int)
    
    df['cell_lat'] = (lat_min + df['lat_idx'] * cell_size).round(4)
    df['cell_lon'] = (lon_min + df['lon_idx'] * cell_size).round(4)
    
    # Cell center coordinates (useful for mapping and distance calculations)
    df['cell_lat_center'] = (df['cell_lat'] + cell_size / 2.0).round(4)
    df['cell_lon_center'] = (df['cell_lon'] + cell_size / 2.0).round(4)
    
    # Unique Cell ID: e.g. "cell_08.10_074.40"
    df['cell_id'] = (
        "cell_" + df['cell_lat'].map(lambda x: f"{x:05.2f}") +
        "_" + df['cell_lon'].map(lambda x: f"{x:06.2f}")
    )
    return df


def build_daily_spatiotemporal_grid(df_points, lat_min=LAT_MIN, lat_max=LAT_MAX,
                                   lon_min=LON_MIN, lon_max=LON_MAX, cell_size=CELL_SIZE):
    """
    Aggregate point detections to daily cell statistics and construct a complete
    spatio-temporal lattice containing both active fire cells and zero-fire cells (negatives).
    
    Parameters:
    -----------
    df_points : pd.DataFrame
        Cleaned point fire observations with assigned grid cells.
        
    Returns:
    --------
    pd.DataFrame
        Complete daily grid dataset containing positive and negative samples.
    """
    print("[Data Prep] Aggregating positive fire detections per cell-day...")
    
    # Point-level indicators for aggregation
    df_points['is_day'] = (df_points['daynight'] == 'D').astype(int)
    df_points['is_high_conf'] = (df_points['confidence'] == 'h').astype(int)

    # Aggregations
    pos_agg = df_points.groupby(['date', 'cell_id']).agg(
        fire_count=('latitude', 'count'),
        mean_frp=('frp', 'mean'),
        max_frp=('frp', 'max'),
        mean_brightness=('brightness', 'mean'),
        mean_bright_t31=('bright_t31', 'mean'),
        day_count=('is_day', 'sum'),
        high_conf_count=('is_high_conf', 'sum'),
        cell_lat=('cell_lat', 'first'),
        cell_lon=('cell_lon', 'first'),
        cell_lat_center=('cell_lat_center', 'first'),
        cell_lon_center=('cell_lon_center', 'first'),
        lat_idx=('lat_idx', 'first'),
        lon_idx=('lon_idx', 'first')
    ).reset_index()

    pos_agg['day_night_ratio'] = (pos_agg['day_count'] / pos_agg['fire_count']).round(4)
    pos_agg['high_conf_share'] = (pos_agg['high_conf_count'] / pos_agg['fire_count']).round(4)
    pos_agg.drop(columns=['day_count', 'high_conf_count'], inplace=True)
    pos_agg['has_fire'] = 1

    print(f"[Data Prep] Aggregated {len(pos_agg):,} active cell-day events.")

    # Generate complete grid definitions across the region
    n_lat_cells = int(np.ceil((lat_max - lat_min) / cell_size))
    n_lon_cells = int(np.ceil((lon_max - lon_min) / cell_size))

    all_cells = []
    for l_idx in range(n_lat_cells):
        c_lat = round(lat_min + l_idx * cell_size, 4)
        c_lat_center = round(c_lat + cell_size / 2.0, 4)
        for lo_idx in range(n_lon_cells):
            c_lon = round(lon_min + lo_idx * cell_size, 4)
            c_lon_center = round(c_lon + cell_size / 2.0, 4)
            c_id = f"cell_{c_lat:05.2f}_{c_lon:06.2f}"
            all_cells.append({
                'cell_id': c_id,
                'cell_lat': c_lat,
                'cell_lon': c_lon,
                'cell_lat_center': c_lat_center,
                'cell_lon_center': c_lon_center,
                'lat_idx': l_idx,
                'lon_idx': lo_idx
            })
    df_all_cells = pd.DataFrame(all_cells)
    total_grid_cells = len(df_all_cells)
    print(f"[Data Prep] Created {total_grid_cells} spatial grid cells ({n_lat_cells} lat x {n_lon_cells} lon).")

    # Generate continuous date range
    min_date = df_points['date'].min()
    max_date = df_points['date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    total_days = len(all_dates)
    print(f"[Data Prep] Time span: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({total_days} days).")

    # Create full Cartesian product (dates x cells)
    idx = pd.MultiIndex.from_product([all_dates, df_all_cells['cell_id']], names=['date', 'cell_id'])
    full_grid = pd.DataFrame(index=idx).reset_index()
    print(f"[Data Prep] Total spatio-temporal lattice points: {len(full_grid):,} (includes positive & zero-fire negatives).")

    # Merge cell metadata
    full_grid = full_grid.merge(df_all_cells, on='cell_id', how='left')

    # Merge positive events
    merged = full_grid.merge(
        pos_agg[['date', 'cell_id', 'fire_count', 'mean_frp', 'max_frp',
                 'mean_brightness', 'mean_bright_t31', 'day_night_ratio', 'high_conf_share', 'has_fire']],
        on=['date', 'cell_id'],
        how='left'
    )

    # Impute zero-fire negatives
    merged['fire_count'] = merged['fire_count'].fillna(0).astype(int)
    merged['mean_frp'] = merged['mean_frp'].fillna(0.0).astype(float)
    merged['max_frp'] = merged['max_frp'].fillna(0.0).astype(float)
    merged['mean_brightness'] = merged['mean_brightness'].fillna(0.0).astype(float)
    merged['mean_bright_t31'] = merged['mean_bright_t31'].fillna(0.0).astype(float)
    merged['day_night_ratio'] = merged['day_night_ratio'].fillna(0.0).astype(float)
    merged['high_conf_share'] = merged['high_conf_share'].fillna(0.0).astype(float)
    merged['has_fire'] = merged['has_fire'].fillna(0).astype(int)

    # Sort deterministically
    merged = merged.sort_values(['cell_id', 'date']).reset_index(drop=True)

    pos_ratio = (merged['has_fire'] == 1).mean() * 100
    print(f"[Data Prep] Lattice ready. Fire occurrences: {merged['has_fire'].sum():,} ({pos_ratio:.2f}%), Zero-fire cells: {(merged['has_fire'] == 0).sum():,} ({100-pos_ratio:.2f}%).")
    
    return merged


def run_data_preparation():
    """Main pipeline execution for data ingestion and gridding."""
    print("=" * 70)
    print("RUNNING STEP 1: DATA PREPARATION & SPATIO-TEMPORAL GRIDDING")
    print("=" * 70)
    
    # 1. Load and combine raw data
    df_raw = load_and_combine_data()
    
    # 2. Assign grid cells
    df_gridded = assign_grid_cells(df_raw)
    
    # Save cleaned point detections
    PROCESSED_CLEANED_POINTS.parent.mkdir(parents=True, exist_ok=True)
    df_gridded.to_parquet(PROCESSED_CLEANED_POINTS, index=False)
    print(f"[Data Prep] Saved cleaned point detections to {PROCESSED_CLEANED_POINTS}")

    # 3. Build complete spatio-temporal lattice with negatives
    df_lattice = build_daily_spatiotemporal_grid(df_gridded)
    
    # Save daily grid features
    df_lattice.to_parquet(PROCESSED_DAILY_GRID, index=False)
    print(f"[Data Prep] Saved full daily grid lattice to {PROCESSED_DAILY_GRID}")
    print("=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return df_gridded, df_lattice


if __name__ == "__main__":
    run_data_preparation()
