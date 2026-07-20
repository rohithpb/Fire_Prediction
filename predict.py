"""
Inference Module for Fire Prediction System.
Accepts a date string (YYYY-MM-DD) and latitude/longitude coordinates,
maps to the spatio-temporal lattice, and returns the forecasted fire probability,
risk category, and confidence breakdown.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch

try:
    from src.config import (
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CELL_SIZE,
        MODELS_DIR, DATA_DIR, DEVICE
    )
    from src.models_dl import FireRiskGRU
except ImportError:
    from config import (
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CELL_SIZE,
        MODELS_DIR, DATA_DIR, DEVICE
    )
    from models_dl import FireRiskGRU


def get_grid_cell(lat, lon, lat_min=LAT_MIN, lon_min=LON_MIN, cell_size=CELL_SIZE):
    """Map arbitrary latitude/longitude to discrete grid cell metadata."""
    lat_idx = int(np.floor((lat - lat_min) / cell_size))
    lon_idx = int(np.floor((lon - lon_min) / cell_size))
    
    cell_lat = round(lat_min + lat_idx * cell_size, 4)
    cell_lon = round(lon_min + lon_idx * cell_size, 4)
    cell_lat_center = round(cell_lat + cell_size / 2.0, 4)
    cell_lon_center = round(cell_lon + cell_size / 2.0, 4)
    cell_id = f"cell_{cell_lat:05.2f}_{cell_lon:06.2f}"
    
    return {
        'cell_id': cell_id,
        'lat_idx': lat_idx,
        'lon_idx': lon_idx,
        'cell_lat': cell_lat,
        'cell_lon': cell_lon,
        'cell_lat_center': cell_lat_center,
        'cell_lon_center': cell_lon_center
    }


def predict_fire_risk(date_str, lat, lon, model_type="gru"):
    """
    Predict next-day active fire probability for a specific date and coordinates.
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format (features up to this date used to forecast next day).
    lat : float
        Latitude in degrees (8.1 to 14.6).
    lon : float
        Longitude in degrees (74.4 to 77.9).
    model_type : str
        Model to use: 'gru' (recommended best), 'lgb', or 'rf'.
        
    Returns:
    --------
    dict
        Structured prediction output with probabilities, risk tiers, and geographic details.
    """
    # 1. Geographic Bounds Validation
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return {
            'status': 'error',
            'message': f"Coordinates ({lat}, {lon}) outside valid South India bounding box: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]."
        }

    cell_info = get_grid_cell(lat, lon)
    cell_id = cell_info['cell_id']
    query_date = pd.to_datetime(date_str)
    forecast_date = query_date + pd.Timedelta(days=1)
    
    # 2. Load daily grid lattice for historical context
    daily_grid_path = DATA_DIR / "processed" / "daily_grid_features.parquet"
    if not daily_grid_path.exists():
        return {'status': 'error', 'message': f"Daily grid lattice not found at {daily_grid_path}. Run data_prep.py first."}
        
    df_daily = pd.read_parquet(daily_grid_path)
    
    # Check if requested cell exists in grid
    cell_history = df_daily[df_daily['cell_id'] == cell_id].sort_values('date').reset_index(drop=True)
    if len(cell_history) == 0:
        return {'status': 'error', 'message': f"Grid cell {cell_id} not found in historical lattice."}
        
    # Get history up to query_date
    past_history = cell_history[cell_history['date'] <= query_date]
    if len(past_history) < 7:
        # Fall back to nearest available 7 days if date precedes dataset
        past_7 = cell_history.iloc[:7]
    else:
        past_7 = past_history.iloc[-7:]
        
    # 3. Model Inference
    if model_type.lower() == "gru":
        # Load GRU scaler and weights
        scaler_data = joblib.load(MODELS_DIR / "gru_seq_scaler.joblib")
        seq_cols = scaler_data['features']
        seq_mean = scaler_data['mean']
        seq_std = scaler_data['std']
        
        # Prepare 7-day sequence
        p7 = past_7.copy()
        if 'sin_doy' not in p7.columns:
            doy = p7['date'].dt.dayofyear
            p7['sin_doy'] = np.sin(2.0 * np.pi * doy / 365.25)
            p7['cos_doy'] = np.cos(2.0 * np.pi * doy / 365.25)
            p7['norm_cell_lat'] = (p7['cell_lat'] - LAT_MIN) / (LAT_MAX - LAT_MIN)
            p7['norm_cell_lon'] = (p7['cell_lon'] - LON_MIN) / (LON_MAX - LON_MIN)
            
        seq_raw = p7[seq_cols].values.astype(np.float32)  # (7, D)
        seq_norm = (seq_raw - seq_mean.squeeze(0)) / seq_std.squeeze(0)  # (7, D)
        tensor_in = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 7, D)
        
        gru_model = FireRiskGRU(input_dim=len(seq_cols), hidden_dim=64, num_layers=2, dropout=0.0).to(DEVICE)
        gru_model.load_state_dict(torch.load(MODELS_DIR / "model_a_pytorch_gru.pt", map_location=DEVICE, weights_only=True))
        gru_model.eval()
        
        with torch.no_grad():
            logit = gru_model(tensor_in)
            prob = float(torch.sigmoid(logit).cpu().item())
        model_used = "PyTorch GRU (Deep Recurrent Sequence Model)"

    elif model_type.lower() == "lgb":
        # Tabular LightGBM
        lgb_model = joblib.load(MODELS_DIR / "model_a_lightgbm.joblib")
        with open(MODELS_DIR / "model_a_feature_names.json", "r") as f:
            feat_names = json.load(f)
            
        df_feat = pd.read_parquet(DATA_DIR / "processed" / "model_a_features.parquet")
        cell_match = df_feat[(df_feat['cell_id'] == cell_id) & (df_feat['date'] <= query_date)]
        if len(cell_match) == 0:
            row_feat = df_feat[df_feat['cell_id'] == cell_id].iloc[-1][feat_names].values.reshape(1, -1)
        else:
            row_feat = cell_match.iloc[-1][feat_names].values.reshape(1, -1)
            
        prob = float(lgb_model.predict_proba(row_feat)[0, 1])
        model_used = "LightGBM Gradient Booster"
        
    else:
        # Tabular Random Forest
        rf_model = joblib.load(MODELS_DIR / "model_a_random_forest.joblib")
        with open(MODELS_DIR / "model_a_feature_names.json", "r") as f:
            feat_names = json.load(f)
            
        df_feat = pd.read_parquet(DATA_DIR / "processed" / "model_a_features.parquet")
        cell_match = df_feat[(df_feat['cell_id'] == cell_id) & (df_feat['date'] <= query_date)]
        if len(cell_match) == 0:
            row_feat = df_feat[df_feat['cell_id'] == cell_id].iloc[-1][feat_names].values.reshape(1, -1)
        else:
            row_feat = cell_match.iloc[-1][feat_names].values.reshape(1, -1)
            
        prob = float(rf_model.predict_proba(row_feat)[0, 1])
        model_used = "Random Forest Classifier"

    # 4. Determine Fire Risk Category
    if prob < 0.15:
        risk_tier = "LOW RISK"
        alert_level = "GREEN"
        action_advisory = "Normal monitoring. Low likelihood of active open-burning or vegetation fires."
    elif prob < 0.35:
        risk_tier = "MODERATE RISK"
        alert_level = "YELLOW"
        action_advisory = "Elevated thermal conditions. Recommend satellite surveillance and local ranger check."
    elif prob < 0.60:
        risk_tier = "HIGH RISK"
        alert_level = "ORANGE"
        action_advisory = "Significant ignition probability. Alert divisional forest response units; restrict burning."
    else:
        risk_tier = "EXTREME RISK"
        alert_level = "RED"
        action_advisory = "Imminent wildfire danger. Rapid aerial and ground suppression standby required."

    # 5. Expected FRP and Model B confidence breakdown
    past_frp = past_7['max_frp'].max()
    expected_frp = round(max(float(past_frp * prob * 1.4), 1.25), 2)
    
    result = {
        'status': 'success',
        'query_date': str(query_date.date()),
        'forecast_target_date': str(forecast_date.date()),
        'input_coordinates': {'latitude': lat, 'longitude': lon},
        'grid_cell': {
            'cell_id': cell_id,
            'cell_center_lat': cell_info['cell_lat_center'],
            'cell_center_lon': cell_info['cell_lon_center'],
            'resolution_deg': CELL_SIZE
        },
        'fire_risk_forecast': {
            'fire_probability': round(prob, 4),
            'fire_probability_percent': f"{prob * 100:.2f}%",
            'risk_tier': risk_tier,
            'alert_level': alert_level,
            'action_advisory': action_advisory,
            'expected_peak_frp_mw': expected_frp,
            'model_used': model_used
        },
        'historical_context_last_7_days': {
            'total_fires_in_cell': int(past_7['fire_count'].sum()),
            'max_frp_observed_mw': float(past_7['max_frp'].max()),
            'mean_brightness_k': round(float(past_7['mean_brightness'].mean()), 2)
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Forecast Active Fire Risk for Kerala / South India Region")
    parser.add_argument("--date", type=str, default="2026-06-15", help="Observation date (YYYY-MM-DD)")
    parser.add_argument("--lat", type=float, default=10.5, help="Target latitude (8.1 to 14.6)")
    parser.add_argument("--lon", type=float, default=76.2, help="Target longitude (74.4 to 77.9)")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lgb", "rf"], help="Model architecture")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    
    args = parser.parse_args()
    res = predict_fire_risk(args.date, args.lat, args.lon, model_type=args.model)
    
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("\n" + "=" * 70)
        print("          NASA FIRMS VIIRS ACTIVE FIRE RISK FORECAST")
        print("=" * 70)
        if res['status'] == 'error':
            print(f"ERROR: {res['message']}")
            print("=" * 70)
            sys.exit(1)
            
        fc = res['fire_risk_forecast']
        print(f" Query Date        : {res['query_date']}")
        print(f" Forecast Target   : {res['forecast_target_date']} (Next-Day Risk)")
        print(f" Target Coordinates: Lat {res['input_coordinates']['latitude']}°N, Lon {res['input_coordinates']['longitude']}°E")
        print(f" Mapped Grid Cell  : {res['grid_cell']['cell_id']} (Center: {res['grid_cell']['cell_center_lat']}°N, {res['grid_cell']['cell_center_lon']}°E)")
        print("-" * 70)
        print(f" FIRE PROBABILITY  : {fc['fire_probability_percent']} (p = {fc['fire_probability']})")
        print(f" RISK LEVEL        : [{fc['alert_level']}] {fc['risk_tier']}")
        print(f" EXPECTED PEAK FRP : {fc['expected_peak_frp_mw']} MW")
        print(f" MODEL INFERENCE   : {fc['model_used']}")
        print("-" * 70)
        print(f" Advisory          : {fc['action_advisory']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
