"""
Model Training Module for Fire Risk Forecasting (Model A) and Confidence Classification (Model B).
Implements chronological splitting, class weighting, baseline models (Logistic Regression,
Random Forest, XGBoost, LightGBM), PyTorch GRU deep sequence model, and Model B ablation.
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from src.config import (
        RANDOM_SEED, set_seed, DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO, DEVICE
    )
    from src.models_dl import (
        FireRiskGRU, FireSequenceDataset, build_sequence_tensors,
        train_epoch, evaluate_dl_model
    )
except ImportError:
    from config import (
        RANDOM_SEED, set_seed, DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO, DEVICE
    )
    from models_dl import (
        FireRiskGRU, FireSequenceDataset, build_sequence_tensors,
        train_epoch, evaluate_dl_model
    )


def chronological_split(df, date_col='date', train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Split dataframe chronologically into strictly non-overlapping train, val, and test subsets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing temporal dates.
    date_col : str
        Column name holding dates.
        
    Returns:
    --------
    tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame, dict)
        (df_train, df_val, df_test, split_metadata)
    """
    unique_dates = np.sort(df[date_col].unique())
    n_dates = len(unique_dates)
    
    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)
    
    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train:n_train + n_val]
    test_dates = unique_dates[n_train + n_val:]
    
    df_train = df[df[date_col].isin(train_dates)].copy().reset_index(drop=True)
    df_val = df[df[date_col].isin(val_dates)].copy().reset_index(drop=True)
    df_test = df[df[date_col].isin(test_dates)].copy().reset_index(drop=True)
    
    split_info = {
        'total_dates': n_dates,
        'train_dates_range': [str(pd.to_datetime(train_dates[0]).date()), str(pd.to_datetime(train_dates[-1]).date())],
        'val_dates_range': [str(pd.to_datetime(val_dates[0]).date()), str(pd.to_datetime(val_dates[-1]).date())],
        'test_dates_range': [str(pd.to_datetime(test_dates[0]).date()), str(pd.to_datetime(test_dates[-1]).date())],
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'train_pos_rate': float((df_train['target_fire_next_day'] == 1).mean() if 'target_fire_next_day' in df_train else 0.0),
        'val_pos_rate': float((df_val['target_fire_next_day'] == 1).mean() if 'target_fire_next_day' in df_val else 0.0),
        'test_pos_rate': float((df_test['target_fire_next_day'] == 1).mean() if 'target_fire_next_day' in df_test else 0.0),
    }
    
    print(f"[Train] Chronological split summary:")
    print(f"        Train : {split_info['train_dates_range'][0]} to {split_info['train_dates_range'][1]} ({len(df_train):,} samples, pos {split_info['train_pos_rate']*100:.2f}%)")
    print(f"        Val   : {split_info['val_dates_range'][0]} to {split_info['val_dates_range'][1]} ({len(df_val):,} samples, pos {split_info['val_pos_rate']*100:.2f}%)")
    print(f"        Test  : {split_info['test_dates_range'][0]} to {split_info['test_dates_range'][1]} ({len(df_test):,} samples, pos {split_info['test_pos_rate']*100:.2f}%)")
    
    return df_train, df_val, df_test, split_info


def calculate_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculate ROC-AUC, PR-AUC, Precision, Recall, F1, and Confusion Matrix.
    """
    y_pred = (y_prob >= threshold).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0
        
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    # Also calculate optimal F1 threshold
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresh = 0.5
    best_f1 = f1
    for th in thresholds:
        th_pred = (y_prob >= th).astype(int)
        score = f1_score(y_true, th_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(th)
            
    return {
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'optimal_threshold': round(best_thresh, 2),
        'optimal_f1': round(best_f1, 4),
        'confusion_matrix': cm
    }


def train_model_a(df_features, df_daily):
    """
    Train and benchmark all Model A architectures (Logistic Regression, Random Forest,
    XGBoost, LightGBM, PyTorch GRU) using strict chronological evaluation.
    
    Returns:
    --------
    tuple (dict, dict, dict)
        (results_summary, saved_models_dict, test_predictions_dict)
    """
    print("\n" + "=" * 70)
    print("STEP 3A: TRAINING MODEL A (FIRE RISK FORECASTING)")
    print("=" * 70)
    set_seed(RANDOM_SEED)
    
    # 1. Chronological Split
    df_train, df_val, df_test, split_meta = chronological_split(df_features, date_col='date')
    
    # Feature columns for tabular models
    ignore_cols = [
        'date', 'cell_id', 'cell_lat', 'cell_lon', 'cell_lat_center', 'cell_lon_center',
        'lat_idx', 'lon_idx', 'has_fire', 'target_fire_next_day', 'target_frp_next_day'
    ]
    feature_cols = [c for c in df_features.columns if c not in ignore_cols]
    print(f"[Train Model A] Number of tabular features: {len(feature_cols)}")
    
    X_train = df_train[feature_cols].values
    y_train = df_train['target_fire_next_day'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['target_fire_next_day'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['target_fire_next_day'].values
    
    # Imbalance scale factor
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"[Train Model A] Training class balance: {neg_count:,} neg vs {pos_count:,} pos (scale_pos_weight: {scale_pos_weight:.2f})")
    
    # StandardScaler for models requiring normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "model_a_scaler.joblib")
    
    results = {}
    test_preds = {'y_test': y_test.tolist()}
    trained_models = {}

    # -------------------------------------------------------------
    # 1. Logistic Regression Baseline
    # -------------------------------------------------------------
    print("\n[1/5] Training Logistic Regression...")
    t0 = time.time()
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver='lbfgs'
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_val_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
    lr_test_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    results['Logistic Regression'] = {
        'val': calculate_binary_metrics(y_val, lr_val_prob),
        'test': calculate_binary_metrics(y_test, lr_test_prob),
        'train_time_sec': round(time.time() - t0, 2)
    }
    test_preds['Logistic Regression'] = lr_test_prob.tolist()
    trained_models['Logistic Regression'] = lr_model
    joblib.dump(lr_model, MODELS_DIR / "model_a_logistic_regression.joblib")
    print(f"      -> Test ROC-AUC: {results['Logistic Regression']['test']['roc_auc']}, PR-AUC: {results['Logistic Regression']['test']['pr_auc']}, F1: {results['Logistic Regression']['test']['f1']}")

    # -------------------------------------------------------------
    # 2. Random Forest Baseline
    # -------------------------------------------------------------
    print("\n[2/5] Training Random Forest...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced_subsample',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_test_prob = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'val': calculate_binary_metrics(y_val, rf_val_prob),
        'test': calculate_binary_metrics(y_test, rf_test_prob),
        'train_time_sec': round(time.time() - t0, 2),
        'feature_importances': rf_model.feature_importances_.tolist()
    }
    test_preds['Random Forest'] = rf_test_prob.tolist()
    trained_models['Random Forest'] = rf_model
    joblib.dump(rf_model, MODELS_DIR / "model_a_random_forest.joblib")
    print(f"      -> Test ROC-AUC: {results['Random Forest']['test']['roc_auc']}, PR-AUC: {results['Random Forest']['test']['pr_auc']}, F1: {results['Random Forest']['test']['f1']}")

    # -------------------------------------------------------------
    # 3. XGBoost Baseline
    # -------------------------------------------------------------
    print("\n[3/5] Training XGBoost...")
    t0 = time.time()
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric='aucpr',
        n_jobs=-1
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    results['XGBoost'] = {
        'val': calculate_binary_metrics(y_val, xgb_val_prob),
        'test': calculate_binary_metrics(y_test, xgb_test_prob),
        'train_time_sec': round(time.time() - t0, 2),
        'feature_importances': xgb_model.feature_importances_.tolist()
    }
    test_preds['XGBoost'] = xgb_test_prob.tolist()
    trained_models['XGBoost'] = xgb_model
    joblib.dump(xgb_model, MODELS_DIR / "model_a_xgboost.joblib")
    print(f"      -> Test ROC-AUC: {results['XGBoost']['test']['roc_auc']}, PR-AUC: {results['XGBoost']['test']['pr_auc']}, F1: {results['XGBoost']['test']['f1']}")

    # -------------------------------------------------------------
    # 4. LightGBM Baseline
    # -------------------------------------------------------------
    print("\n[4/5] Training LightGBM...")
    t0 = time.time()
    lgb_model = LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[]
    )
    lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]
    
    results['LightGBM'] = {
        'val': calculate_binary_metrics(y_val, lgb_val_prob),
        'test': calculate_binary_metrics(y_test, lgb_test_prob),
        'train_time_sec': round(time.time() - t0, 2),
        'feature_importances': lgb_model.feature_importances_.tolist()
    }
    test_preds['LightGBM'] = lgb_test_prob.tolist()
    trained_models['LightGBM'] = lgb_model
    joblib.dump(lgb_model, MODELS_DIR / "model_a_lightgbm.joblib")
    print(f"      -> Test ROC-AUC: {results['LightGBM']['test']['roc_auc']}, PR-AUC: {results['LightGBM']['test']['pr_auc']}, F1: {results['LightGBM']['test']['f1']}")

    # -------------------------------------------------------------
    # 5. Deep Sequence Model: PyTorch GRU
    # -------------------------------------------------------------
    print("\n[5/5] Training PyTorch GRU Recurrent Sequence Model...")
    t0 = time.time()
    
    # Build 7-day sequence tensor vectorised across daily grid
    X_seq_all, y_seq_all, seq_cols = build_sequence_tensors(df_daily, df_features, seq_len=7)
    
    # Split sequence tensors using the exact same split indices
    train_mask = df_features['date'].isin(df_train['date'].unique())
    val_mask = df_features['date'].isin(df_val['date'].unique())
    test_mask = df_features['date'].isin(df_test['date'].unique())
    
    X_seq_train, y_seq_train = X_seq_all[train_mask], y_seq_all[train_mask]
    X_seq_val, y_seq_val = X_seq_all[val_mask], y_seq_all[val_mask]
    X_seq_test, y_seq_test = X_seq_all[test_mask], y_seq_all[test_mask]
    
    # Standardize sequence features per step across dataset
    seq_mean = X_seq_train.mean(axis=(0, 1), keepdims=True)
    seq_std = X_seq_train.std(axis=(0, 1), keepdims=True) + 1e-6
    
    X_seq_train = (X_seq_train - seq_mean) / seq_std
    X_seq_val = (X_seq_val - seq_mean) / seq_std
    X_seq_test = (X_seq_test - seq_mean) / seq_std
    
    # Save sequence scaling parameters
    joblib.dump({'mean': seq_mean, 'std': seq_std, 'features': seq_cols}, MODELS_DIR / "gru_seq_scaler.joblib")

    # Dataloaders
    batch_size = 256
    train_ds = FireSequenceDataset(X_seq_train, y_seq_train)
    val_ds = FireSequenceDataset(X_seq_val, y_seq_val)
    test_ds = FireSequenceDataset(X_seq_test, y_seq_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize GRU
    gru_model = FireRiskGRU(
        input_dim=len(seq_cols),
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    ).to(DEVICE)
    
    # Weighted BCE loss for severe imbalance
    pos_weight_tensor = torch.tensor([scale_pos_weight], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(gru_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training Loop with Early Stopping
    epochs = 12
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': [], 'val_pr_auc': []}
    best_val_auc = 0.0
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_auc, tr_prauc = train_epoch(gru_model, train_loader, criterion, optimizer, device=DEVICE)
        vl_loss, vl_auc, vl_prauc, _, _ = evaluate_dl_model(gru_model, val_loader, criterion, device=DEVICE)
        scheduler.step(vl_auc)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(round(tr_loss, 4))
        history['val_loss'].append(round(vl_loss, 4))
        history['train_auc'].append(round(tr_auc, 4))
        history['val_auc'].append(round(vl_auc, 4))
        history['val_pr_auc'].append(round(vl_prauc, 4))
        
        print(f"      Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f}, Train AUC: {tr_auc:.4f} | Val Loss: {vl_loss:.4f}, Val AUC: {vl_auc:.4f}, Val PR-AUC: {vl_prauc:.4f}")
        
        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            best_weights = gru_model.state_dict().copy()
            
    # Load best weights
    if best_weights is not None:
        gru_model.load_state_dict(best_weights)
        
    # Evaluate GRU on test set
    _, _, _, gru_val_prob, _ = evaluate_dl_model(gru_model, val_loader, criterion, device=DEVICE)
    _, _, _, gru_test_prob, _ = evaluate_dl_model(gru_model, test_loader, criterion, device=DEVICE)
    
    results['PyTorch GRU'] = {
        'val': calculate_binary_metrics(y_val, gru_val_prob),
        'test': calculate_binary_metrics(y_test, gru_test_prob),
        'train_time_sec': round(time.time() - t0, 2),
        'training_history': history
    }
    test_preds['PyTorch GRU'] = gru_test_prob.tolist()
    trained_models['PyTorch GRU'] = gru_model
    torch.save(gru_model.state_dict(), MODELS_DIR / "model_a_pytorch_gru.pt")
    print(f"      -> Test ROC-AUC: {results['PyTorch GRU']['test']['roc_auc']}, PR-AUC: {results['PyTorch GRU']['test']['pr_auc']}, F1: {results['PyTorch GRU']['test']['f1']}")

    # Save feature columns metadata
    with open(MODELS_DIR / "model_a_feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    return results, trained_models, test_preds, feature_cols, split_meta


def train_model_b(df_points):
    """
    Train Model B (Fire Confidence Multi-Class Classification: Low, Nominal, High)
    and conduct the ablation experiment (with vs without brightness).
    
    Returns:
    --------
    tuple (dict, dict)
        (results_summary, saved_models_dict)
    """
    print("\n" + "=" * 70)
    print("STEP 3B: TRAINING MODEL B (CONFIDENCE CLASSIFICATION & ABLATION)")
    print("=" * 70)
    set_seed(RANDOM_SEED)
    
    # Features definitions
    full_cols = [
        'brightness', 'bright_t31', 'frp', 'scan', 'track',
        'is_day', 'latitude', 'longitude', 'hour', 'month'
    ]
    no_brightness_cols = [
        'frp', 'scan', 'track',
        'is_day', 'latitude', 'longitude', 'hour', 'month'
    ]
    
    # Chronological split on point detections
    df_sorted = df_points.sort_values('acq_datetime').reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    train_b = df_sorted.iloc[:n_train]
    val_b = df_sorted.iloc[n_train:n_train + n_val]
    test_b = df_sorted.iloc[n_train + n_val:]
    
    y_train = train_b['target_confidence'].values
    y_val = val_b['target_confidence'].values
    y_test = test_b['target_confidence'].values
    
    results = {}
    models_b = {}
    
    # -------------------------------------------------------------
    # Experiment 1: With Brightness Features
    # -------------------------------------------------------------
    print("\n[Model B - Exp 1] Training with ALL Features (including Brightness)...")
    X_train_full = train_b[full_cols].values
    X_val_full = val_b[full_cols].values
    X_test_full = test_b[full_cols].values
    
    rf_b_full = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_b_full.fit(X_train_full, y_train)
    
    y_pred_test_full = rf_b_full.predict(X_test_full)
    y_prob_test_full = rf_b_full.predict_proba(X_test_full)
    
    macro_f1_full = float(f1_score(y_test, y_pred_test_full, average='macro'))
    weighted_f1_full = float(f1_score(y_test, y_pred_test_full, average='weighted'))
    cm_full = confusion_matrix(y_test, y_pred_test_full).tolist()
    
    results['Model B (With Brightness)'] = {
        'macro_f1': round(macro_f1_full, 4),
        'weighted_f1': round(weighted_f1_full, 4),
        'confusion_matrix': cm_full,
        'feature_names': full_cols,
        'feature_importances': rf_b_full.feature_importances_.tolist(),
        'classification_report': classification_report(y_test, y_pred_test_full, output_dict=True, zero_division=0)
    }
    models_b['full'] = rf_b_full
    joblib.dump(rf_b_full, MODELS_DIR / "model_b_with_brightness.joblib")
    print(f"      -> Macro-F1: {macro_f1_full:.4f}, Weighted-F1: {weighted_f1_full:.4f}")

    # -------------------------------------------------------------
    # Experiment 2: Ablation with Brightness Removed
    # -------------------------------------------------------------
    print("\n[Model B - Exp 2] Ablation Experiment (BRIGHTNESS REMOVED)...")
    X_train_no_br = train_b[no_brightness_cols].values
    X_val_no_br = val_b[no_brightness_cols].values
    X_test_no_br = test_b[no_brightness_cols].values
    
    rf_b_no_br = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_b_no_br.fit(X_train_no_br, y_train)
    
    y_pred_test_no_br = rf_b_no_br.predict(X_test_no_br)
    macro_f1_no_br = float(f1_score(y_test, y_pred_test_no_br, average='macro'))
    weighted_f1_no_br = float(f1_score(y_test, y_pred_test_no_br, average='weighted'))
    cm_no_br = confusion_matrix(y_test, y_pred_test_no_br).tolist()
    
    results['Model B (Ablation - No Brightness)'] = {
        'macro_f1': round(macro_f1_no_br, 4),
        'weighted_f1': round(weighted_f1_no_br, 4),
        'confusion_matrix': cm_no_br,
        'feature_names': no_brightness_cols,
        'feature_importances': rf_b_no_br.feature_importances_.tolist(),
        'classification_report': classification_report(y_test, y_pred_test_no_br, output_dict=True, zero_division=0)
    }
    models_b['no_brightness'] = rf_b_no_br
    joblib.dump(rf_b_no_br, MODELS_DIR / "model_b_no_brightness.joblib")
    print(f"      -> Macro-F1: {macro_f1_no_br:.4f}, Weighted-F1: {weighted_f1_no_br:.4f}")
    print(f"      -> Impact of removing brightness: Macro-F1 delta = {macro_f1_no_br - macro_f1_full:+.4f}")

    return results, models_b, (y_test, y_pred_test_full, y_pred_test_no_br)


def run_training_pipeline():
    """Main orchestrator for training and saving all models."""
    print("=" * 70)
    print("STARTING FULL TRAINING PIPELINE")
    print("=" * 70)
    
    # Load prepared datasets
    model_a_path = DATA_DIR / "processed" / "model_a_features.parquet"
    daily_path = DATA_DIR / "processed" / "daily_grid_features.parquet"
    model_b_path = DATA_DIR / "processed" / "model_b_features.parquet"
    
    df_features = pd.read_parquet(model_a_path)
    df_daily = pd.read_parquet(daily_path)
    df_points = pd.read_parquet(model_b_path)
    
    # 1. Train Model A
    results_a, models_a, test_preds_a, feature_cols, split_meta = train_model_a(df_features, df_daily)
    
    # 2. Train Model B
    results_b, models_b, test_preds_b = train_model_b(df_points)
    
    # Save combined metrics
    all_metrics = {
        'model_a': results_a,
        'model_b': results_b,
        'split_metadata': split_meta
    }
    
    metrics_file = OUTPUTS_DIR / "metrics_summary.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Train] Saved complete metrics summary to {metrics_file}")
    
    # Save test predictions for evaluation and plotting
    test_preds_file = OUTPUTS_DIR / "test_predictions.joblib"
    joblib.dump({
        'model_a': test_preds_a,
        'model_b': {
            'y_test': test_preds_b[0].tolist(),
            'y_pred_full': test_preds_b[1].tolist(),
            'y_pred_no_br': test_preds_b[2].tolist()
        }
    }, test_preds_file)
    print(f"[Train] Saved predictions cache to {test_preds_file}")
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return all_metrics


if __name__ == "__main__":
    run_training_pipeline()
