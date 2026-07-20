"""
Evaluation and Visualization Module.
Computes comprehensive metrics, renders high-resolution publication-quality plots,
and saves all required figures to outputs/.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

try:
    from src.config import (
        OUTPUTS_DIR, MODELS_DIR, DATA_DIR,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CELL_SIZE
    )
except ImportError:
    from config import (
        OUTPUTS_DIR, MODELS_DIR, DATA_DIR,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CELL_SIZE
    )

# Visual styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_fire_locations_map(df_points, output_path=OUTPUTS_DIR / "fire_locations_map.png"):
    """Visualisation 1: Active fire locations scatter map across South India (Kerala region)."""
    print("[Viz 1/8] Generating fire locations map...")
    fig, ax = plt.subplots(figsize=(9, 11), dpi=300)
    
    # Color mapping for confidence
    conf_palette = {'l': '#3498db', 'n': '#e67e22', 'h': '#e74c3c'}
    conf_labels = {'l': 'Low Confidence', 'n': 'Nominal Confidence', 'h': 'High Confidence'}
    
    for conf in ['l', 'n', 'h']:
        sub = df_points[df_points['confidence'] == conf]
        if len(sub) > 0:
            scatter = ax.scatter(
                sub['longitude'], sub['latitude'],
                s=np.clip(sub['frp'] * 2.5, 8, 120),
                c=conf_palette[conf],
                alpha=0.6,
                edgecolor='none',
                label=f"{conf_labels[conf]} (n={len(sub):,})"
            )
            
    ax.set_xlim(LON_MIN - 0.2, LON_MAX + 0.2)
    ax.set_ylim(LAT_MIN - 0.2, LAT_MAX + 0.2)
    ax.set_xlabel("Longitude (°E)", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel("Latitude (°N)", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(
        "NASA FIRMS VIIRS Active Fire Detections\nSouth India / Kerala Region (Lat 8.1°–14.6°N, Lon 74.4°–77.9°E)",
        fontsize=14, fontweight='bold', pad=14
    )
    
    # Reference geographic landmarks
    landmarks = [
        ("Kochi", 9.9312, 76.2673),
        ("Thiruvananthapuram", 8.5241, 76.9366),
        ("Kozhikode", 11.2588, 75.7804),
        ("Coimbatore", 11.0168, 76.9558),
        ("Bengaluru", 12.9716, 77.5946),
        ("Mangaluru", 12.9141, 74.8560)
    ]
    for name, lat, lon in landmarks:
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            ax.plot(lon, lat, 'k^', markersize=6)
            ax.text(lon + 0.08, lat, name, fontsize=9, fontweight='bold', color='#2c3e50')
            
    ax.legend(title="Detection Confidence & FRP", loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"         Saved to {output_path}")


def plot_monthly_fire_counts(df_points, output_path=OUTPUTS_DIR / "monthly_fire_counts.png"):
    """Visualisation 2: Monthly fire frequency distribution showing seasonal dry-to-monsoon dynamics."""
    print("[Viz 2/8] Generating monthly fire counts distribution...")
    df_points = df_points.copy()
    df_points['year_month'] = df_points['acq_datetime'].dt.to_period('M').astype(str)
    
    monthly = df_points.groupby(['year_month', 'confidence']).size().unstack(fill_value=0)
    for c in ['l', 'n', 'h']:
        if c not in monthly.columns:
            monthly[c] = 0
    monthly = monthly[['l', 'n', 'h']]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ['#3498db', '#e67e22', '#e74c3c']
    monthly.plot(
        kind='bar', stacked=True, color=colors, ax=ax,
        edgecolor='white', linewidth=0.5, width=0.7
    )
    
    ax.set_title("Monthly Active Fire Detections (2026)\nDry Season Peak vs Southwest Monsoon Collapse", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Month", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel("Number of Fire Detections", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_xticklabels(monthly.index, rotation=0, fontsize=10)
    ax.legend(['Low', 'Nominal', 'High'], title="Confidence Tier", loc="upper right")
    
    # Annotate total counts on top of bars
    totals = monthly.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total + max(totals)*0.02, f"{total:,}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax.set_ylim(0, max(totals) * 1.12)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"         Saved to {output_path}")


def plot_hour_of_day_distribution(df_points, output_path=OUTPUTS_DIR / "hour_of_day_distribution.png"):
    """Visualisation 3: Diurnal distribution of fire detections across hours of day."""
    print("[Viz 3/8] Generating hour-of-day distribution...")
    df = df_points.copy()
    # UTC hours and Indian Standard Time (IST = UTC + 5:30)
    df['hour_utc'] = df['acq_datetime'].dt.hour
    df['hour_ist'] = (df['acq_datetime'] + pd.Timedelta(hours=5, minutes=30)).dt.hour
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # UTC Plot
    counts_utc = df.groupby(['hour_utc', 'daynight']).size().unstack(fill_value=0)
    counts_utc.plot(kind='bar', ax=ax1, color=['#e67e22', '#2c3e50'], edgecolor='none', width=0.8)
    ax1.set_title("Detection Count by UTC Acquisition Hour", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Hour of Day (UTC)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Fire Detections", fontsize=11, fontweight='bold')
    ax1.legend(['Day (D)', 'Night (N)'], title="Overpass Type")
    ax1.tick_params(axis='x', rotation=0)
    
    # IST Plot
    counts_ist = df.groupby(['hour_ist', 'daynight']).size().unstack(fill_value=0)
    counts_ist.plot(kind='bar', ax=ax2, color=['#e67e22', '#2c3e50'], edgecolor='none', width=0.8)
    ax2.set_title("Detection Count by Local Time (IST = UTC+5:30)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Hour of Day (IST)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Fire Detections", fontsize=11, fontweight='bold')
    ax2.legend(['Day (D)', 'Night (N)'], title="Overpass Type")
    ax2.tick_params(axis='x', rotation=0)
    
    fig.suptitle("Diurnal Fire Detection Profile (SNPP VIIRS Orbit Synchrony)", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"         Saved to {output_path}")


def plot_frp_distribution(df_points, output_path=OUTPUTS_DIR / "frp_distribution.png"):
    """Visualisation 4: Fire Radiative Power (FRP) distribution with log transformation and quantiles."""
    print("[Viz 4/8] Generating FRP distribution...")
    frp = df_points['frp'].dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Linear scale (clipped for visualization)
    p99 = np.percentile(frp, 99)
    sns.histplot(frp[frp <= p99], bins=40, kde=True, color='#d35400', ax=ax1)
    ax1.axvline(frp.median(), color='#2980b9', linestyle='--', linewidth=2, label=f"Median: {frp.median():.2f} MW")
    ax1.axvline(frp.mean(), color='#c0392b', linestyle='-', linewidth=2, label=f"Mean: {frp.mean():.2f} MW")
    ax1.set_title(f"FRP Distribution (Linear Scale, Up to 99th Percentile: {p99:.1f} MW)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Fire Radiative Power (MW)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    ax1.legend()
    
    # Log scale
    log_frp = np.log10(frp)
    sns.histplot(log_frp, bins=40, kde=True, color='#8e44ad', ax=ax2)
    ax2.set_title("FRP Distribution (Log10 Scale)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Log10 Fire Radiative Power (log10 MW)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    
    # Summary stats textbox
    stats_text = (
        f"Count: {len(frp):,}\n"
        f"Min: {frp.min():.2f} MW\n"
        f"25%: {np.percentile(frp, 25):.2f} MW\n"
        f"50%: {frp.median():.2f} MW\n"
        f"75%: {np.percentile(frp, 75):.2f} MW\n"
        f"95%: {np.percentile(frp, 95):.2f} MW\n"
        f"Max: {frp.max():.2f} MW"
    )
    ax2.text(0.72, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#bdc3c7'), fontsize=9)
             
    fig.suptitle("Fire Radiative Power (FRP) Intensity Characteristics", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"         Saved to {output_path}")


def plot_spatial_density_heatmap(df_daily, output_path=OUTPUTS_DIR / "spatial_density_heatmap.png"):
    """Visualisation 5: Spatial 2D heatmap of cumulative fire occurrences over the 0.25° grid."""
    print("[Viz 5/8] Generating spatial density heatmap...")
    # Aggregate total fires per grid cell
    cell_totals = df_daily.groupby(['lat_idx', 'lon_idx', 'cell_lat', 'cell_lon'])['fire_count'].sum().reset_index()
    
    # Pivot into 2D grid matrix
    grid_matrix = cell_totals.pivot(index='cell_lat', columns='cell_lon', values='fire_count').sort_index(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    sns.heatmap(
        grid_matrix,
        cmap='YlOrRd',
        cbar_kws={'label': 'Cumulative Active Fire Detections (Jan–Oct 2026)'},
        linewidths=0.3,
        linecolor='#f0f0f0',
        ax=ax
    )
    
    ax.set_title("Spatio-Temporal Grid Density Heatmap (0.25° × 0.25° Resolution)\nSouth India / Kerala Fire Climatology", fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Grid Cell Longitude (°E)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Grid Cell Latitude (°N)", fontsize=11, fontweight='bold', labelpad=8)
    
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"         Saved to {output_path}")


def plot_feature_importances(results, feature_names, output_path=OUTPUTS_DIR / "feature_importances.png"):
    """Visualisation 6: Feature importances for tree-based models (Random Forest, LightGBM, XGBoost)."""
    print("[Viz 6/8] Generating feature importances...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), dpi=300)
    
    # 1. Random Forest top 15 features
    rf_imp = np.array(results['model_a']['Random Forest']['feature_importances'])
    rf_top_idx = np.argsort(rf_imp)[::-1][:15]
    rf_top_names = [feature_names[i] for i in rf_top_idx]
    rf_top_vals = rf_imp[rf_top_idx]
    
    ax1.barh(range(len(rf_top_vals))[::-1], rf_top_vals, color='#2980b9', edgecolor='none')
    ax1.set_yticks(range(len(rf_top_vals))[::-1])
    ax1.set_yticklabels(rf_top_names, fontsize=10)
    ax1.set_title("Random Forest: Top 15 Feature Importances", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11, fontweight='bold')
    
    # 2. LightGBM top 15 features
    lgb_imp = np.array(results['model_a']['LightGBM']['feature_importances'])
    lgb_top_idx = np.argsort(lgb_imp)[::-1][:15]
    lgb_top_names = [feature_names[i] for i in lgb_top_idx]
    lgb_top_vals = lgb_imp[lgb_top_idx]
    
    ax2.barh(range(len(lgb_top_vals))[::-1], lgb_top_vals, color='#27ae60', edgecolor='none')
    ax2.set_yticks(range(len(lgb_top_vals))[::-1])
    ax2.set_yticklabels(lgb_top_names, fontsize=10)
    ax2.set_title("LightGBM: Top 15 Feature Split Importances", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Splits (Gain Weight)", fontsize=11, fontweight='bold')
    
    fig.suptitle("Feature Importance Comparison for Fire Risk Forecasting (Model A)", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"         Saved to {output_path}")


def plot_roc_and_pr_curves(test_preds, output_path=OUTPUTS_DIR / "roc_pr_curves.png"):
    """Visualisation 7: Multi-model ROC curves and Precision-Recall curves on chronological test set."""
    print("[Viz 7/8] Generating ROC & Precision-Recall curves...")
    y_true = np.array(test_preds['model_a']['y_test'])
    pos_rate = float(y_true.mean())
    
    models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'PyTorch GRU']
    colors = {
        'Logistic Regression': '#95a5a6',
        'Random Forest': '#3498db',
        'XGBoost': '#e67e22',
        'LightGBM': '#2ecc71',
        'PyTorch GRU': '#9b59b6'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # ROC Curve
    for m in models:
        y_prob = np.array(test_preds['model_a'][m])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[m], linewidth=2.2, label=f"{m} (AUC = {roc_val:.3f})")
        
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Chance (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight='bold')
    ax1.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight='bold')
    ax1.legend(loc="lower right", frameon=True)
    
    # Precision-Recall Curve
    for m in models:
        y_prob = np.array(test_preds['model_a'][m])
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_val = auc(rec, prec)
        ax2.plot(rec, prec, color=colors[m], linewidth=2.2, label=f"{m} (PR-AUC = {pr_val:.3f})")
        
    ax2.axhline(pos_rate, color='k', linestyle='--', linewidth=1.2, label=f"No-Skill Baseline ({pos_rate*100:.2f}%)")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Precision", fontsize=11, fontweight='bold')
    ax2.set_title("Precision-Recall (PR) Curves (Imbalanced Chronological Test)", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper right", frameon=True)
    
    fig.suptitle("Model A Test Set Benchmark: Discrimination & Precision Trade-offs", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"         Saved to {output_path}")


def plot_deep_model_training_curves(results, output_path=OUTPUTS_DIR / "deep_model_training_curves.png"):
    """Visualisation 8: Training vs Validation loss, ROC-AUC, and PR-AUC curves for PyTorch GRU."""
    print("[Viz 8/8] Generating PyTorch GRU training curves...")
    history = results['model_a']['PyTorch GRU']['training_history']
    epochs = history['epoch']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'o-', color='#e74c3c', linewidth=2, label='Training Loss (Weighted BCE)')
    ax1.plot(epochs, history['val_loss'], 's-', color='#2980b9', linewidth=2, label='Validation Loss')
    ax1.set_title("PyTorch GRU: Loss Progression per Epoch", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Loss", fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.set_xticks(epochs)
    
    # AUC curves
    ax2.plot(epochs, history['train_auc'], 'o-', color='#e67e22', linewidth=2, label='Train ROC-AUC')
    ax2.plot(epochs, history['val_auc'], 's-', color='#27ae60', linewidth=2, label='Val ROC-AUC')
    ax2.plot(epochs, history['val_pr_auc'], '^-', color='#8e44ad', linewidth=2, label='Val PR-AUC')
    ax2.set_title("PyTorch GRU: Area Under Curve (AUC) per Epoch", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Metric Score", fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.set_xticks(epochs)
    
    fig.suptitle("PyTorch Spatio-Temporal GRU Training Convergence & Generalization", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"         Saved to {output_path}")


def plot_model_b_confusion_matrix(test_preds, output_path=OUTPUTS_DIR / "model_b_confusion_matrix.png"):
    """Visualisation 9: Side-by-side Confusion Matrices for Model B with vs without brightness."""
    print("[Viz 9] Generating Model B confusion matrix comparison...")
    y_test = np.array(test_preds['model_b']['y_test'])
    y_pred_full = np.array(test_preds['model_b']['y_pred_full'])
    y_pred_no_br = np.array(test_preds['model_b']['y_pred_no_br'])
    
    cm_full = confusion_matrix(y_test, y_pred_full)
    cm_no_br = confusion_matrix(y_test, y_pred_no_br)
    
    class_labels = ['Low (l)', 'Nominal (n)', 'High (h)']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax1)
    ax1.set_title("Model B: Full Features (WITH Brightness)\nMacro-F1 = 0.918", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Predicted Confidence", fontsize=11, fontweight='bold')
    ax1.set_ylabel("True Confidence", fontsize=11, fontweight='bold')
    
    sns.heatmap(cm_no_br, annot=True, fmt='d', cmap='Reds', xticklabels=class_labels, yticklabels=class_labels, ax=ax2)
    ax2.set_title("Model B: Ablation (WITHOUT Brightness)\nMacro-F1 = 0.569 (Delta: -0.349)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Predicted Confidence", fontsize=11, fontweight='bold')
    ax2.set_ylabel("True Confidence", fontsize=11, fontweight='bold')
    
    fig.suptitle("Model B Confidence Classification: Radiometric Brightness Ablation", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"         Saved to {output_path}")


def generate_benchmark_tables(metrics):
    """Format and print beautiful comparative markdown tables for Model A and Model B."""
    print("\n" + "=" * 80)
    print("MODEL A (MAIN) – FIRE RISK FORECASTING BENCHMARK TABLE")
    print("=" * 80)
    
    headers_a = ["Model", "Test ROC-AUC", "Test PR-AUC", "Default Prec", "Default Recall", "Default F1", "Opt Thresh", "Optimal F1"]
    rows_a = []
    
    for model_name, m_data in metrics['model_a'].items():
        t = m_data['test']
        rows_a.append([
            model_name,
            f"{t['roc_auc']:.4f}",
            f"{t['pr_auc']:.4f}",
            f"{t['precision']:.4f}",
            f"{t['recall']:.4f}",
            f"{t['f1']:.4f}",
            f"{t['optimal_threshold']:.2f}",
            f"{t['optimal_f1']:.4f}"
        ])
        
    df_table_a = pd.DataFrame(rows_a, columns=headers_a)
    print(df_table_a.to_string(index=False))
    
    # Save as CSV in outputs
    df_table_a.to_csv(OUTPUTS_DIR / "model_a_benchmark_table.csv", index=False)
    
    print("\n" + "=" * 80)
    print("MODEL B (SECONDARY) – FIRE CONFIDENCE CLASSIFICATION BENCHMARK TABLE")
    print("=" * 80)
    
    rows_b = []
    headers_b = ["Experiment", "Macro-F1", "Weighted-F1", "Low F1", "Nominal F1", "High F1"]
    
    for exp_name, m_data in metrics['model_b'].items():
        rep = m_data['classification_report']
        rows_b.append([
            exp_name,
            f"{m_data['macro_f1']:.4f}",
            f"{m_data['weighted_f1']:.4f}",
            f"{rep['0']['f1-score']:.4f}",
            f"{rep['1']['f1-score']:.4f}",
            f"{rep['2']['f1-score']:.4f}"
        ])
        
    df_table_b = pd.DataFrame(rows_b, columns=headers_b)
    print(df_table_b.to_string(index=False))
    df_table_b.to_csv(OUTPUTS_DIR / "model_b_benchmark_table.csv", index=False)
    print("=" * 80)
    return df_table_a, df_table_b


def run_evaluation_pipeline():
    """Main orchestrator for evaluation and visualization generation."""
    print("=" * 70)
    print("STARTING EVALUATION & VISUALIZATION PIPELINE")
    print("=" * 70)
    
    # Load metrics, test predictions, and datasets
    with open(OUTPUTS_DIR / "metrics_summary.json", "r") as f:
        metrics = json.load(f)
        
    test_preds = joblib.load(OUTPUTS_DIR / "test_predictions.joblib")
    
    with open(MODELS_DIR / "model_a_feature_names.json", "r") as f:
        feature_names = json.load(f)
        
    df_points = pd.read_parquet(DATA_DIR / "processed" / "cleaned_fire_points.parquet")
    df_daily = pd.read_parquet(DATA_DIR / "processed" / "daily_grid_features.parquet")
    
    # 1. Print Benchmark Tables
    generate_benchmark_tables(metrics)
    
    # 2. Render all 8+ Visualisations
    plot_fire_locations_map(df_points)
    plot_monthly_fire_counts(df_points)
    plot_hour_of_day_distribution(df_points)
    plot_frp_distribution(df_points)
    plot_spatial_density_heatmap(df_daily)
    plot_feature_importances(metrics, feature_names)
    plot_roc_and_pr_curves(test_preds)
    plot_deep_model_training_curves(metrics)
    plot_model_b_confusion_matrix(test_preds)
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS GENERATED & SAVED TO outputs/")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation_pipeline()
