"""
PyTorch Deep Learning Module for Fire Risk Forecasting (Model A).
Implements a 7-day Spatio-Temporal Sequence GRU model with BCEWithLogitsLoss
and positive class weighting for severe class imbalance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from src.config import RANDOM_SEED, DEVICE
except ImportError:
    from config import RANDOM_SEED, DEVICE


class FireSequenceDataset(Dataset):
    """
    PyTorch Dataset providing 7-day historical sequence matrices and target labels.
    """
    def __init__(self, sequences, targets, feature_names=None):
        """
        Parameters:
        -----------
        sequences : np.ndarray
            3D array of shape (N, seq_len=7, num_features).
        targets : np.ndarray
            1D array of shape (N,) containing binary target values.
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        self.feature_names = feature_names

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class FireRiskGRU(nn.Module):
    """
    2-Layer Gated Recurrent Unit (GRU) for Spatio-Temporal Fire Risk Forecasting.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(FireRiskGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Recurrent GRU backbone
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dense classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, seq_len, input_dim)
        """
        gru_out, _ = self.gru(x)
        # Take the final hidden state of the sequence
        last_step = gru_out[:, -1, :]
        logits = self.head(last_step)
        return logits


def train_epoch(model, dataloader, criterion, optimizer, device=DEVICE):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs.flatten())
        all_targets.extend(y_batch.cpu().numpy().flatten())
        
    avg_loss = total_loss / len(dataloader.dataset)
    try:
        roc_auc = roc_auc_score(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds)
    except Exception:
        roc_auc, pr_auc = 0.5, 0.0
        
    return avg_loss, roc_auc, pr_auc


def evaluate_dl_model(model, dataloader, criterion, device=DEVICE):
    """Evaluate deep learning model on validation or test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * len(y_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds)
    except Exception:
        roc_auc, pr_auc = 0.5, 0.0
        
    return avg_loss, roc_auc, pr_auc, all_preds, all_targets


def build_sequence_tensors(df_daily, df_features, seq_len=7):
    """
    Construct 3D sequence tensor (N, seq_len, num_features) vectorised
    using temporal indexing across contiguous daily grid cell records.
    
    Parameters:
    -----------
    df_daily : pd.DataFrame
        Daily lattice dataframe containing base features.
    df_features : pd.DataFrame
        Model A dataset containing targets and matched dates/cells.
    seq_len : int
        Sequence length (default 7 days).
        
    Returns:
    --------
    tuple (np.ndarray, np.ndarray, list of str)
        (X_seq, y_targets, feature_names)
    """
    print(f"[Deep Learning] Vectorising {seq_len}-day temporal sequence tensor...")
    # Add sin/cos doy and normalized coords to df_daily if not present
    daily = df_daily.copy()
    if 'sin_doy' not in daily.columns:
        doy = daily['date'].dt.dayofyear
        daily['sin_doy'] = np.sin(2.0 * np.pi * doy / 365.25)
        daily['cos_doy'] = np.cos(2.0 * np.pi * doy / 365.25)
        daily['norm_cell_lat'] = (daily['cell_lat'] - daily['cell_lat'].min()) / (daily['cell_lat'].max() - daily['cell_lat'].min())
        daily['norm_cell_lon'] = (daily['cell_lon'] - daily['cell_lon'].min()) / (daily['cell_lon'].max() - daily['cell_lon'].min())

    # Daily features to include at each step of the sequence
    seq_feature_cols = [
        'fire_count', 'max_frp', 'mean_brightness', 'mean_bright_t31',
        'sin_doy', 'cos_doy', 'norm_cell_lat', 'norm_cell_lon'
    ]
    
    # Sort strictly by cell_id and date
    daily = daily.sort_values(['cell_id', 'date']).reset_index(drop=True)
    daily['row_idx'] = np.arange(len(daily))
    
    # Map (cell_id, date) to row_idx
    row_idx_map = daily.set_index(['cell_id', 'date'])['row_idx']
    
    # Target rows
    feat_keys = list(zip(df_features['cell_id'], df_features['date']))
    target_indices = row_idx_map.loc[feat_keys].values
    
    # Extract feature matrix
    feat_matrix = daily[seq_feature_cols].values.astype(np.float32)
    
    # Vectorised gather of 7 time steps [t-6, t-5, t-4, t-3, t-2, t-1, t]
    step_matrices = []
    for offset in range(seq_len - 1, -1, -1):
        step_indices = target_indices - offset
        step_matrices.append(feat_matrix[step_indices])
        
    X_seq = np.stack(step_matrices, axis=1)  # (N, seq_len, D)
    y_targets = df_features['target_fire_next_day'].values.astype(np.float32)
    
    print(f"[Deep Learning] Sequence tensor shape: {X_seq.shape}, Targets shape: {y_targets.shape}")
    return X_seq, y_targets, seq_feature_cols
