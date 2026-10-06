"""Feature engineering module for EPIStress signals.
Extracts sensible features for raw modalities (Accelerometer) and organizes
pre-extracted features with standardized modality prefixes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def extract_accelerometer_features(
    acc_signals: Dict[str, pd.Series],
    num_windows: int
) -> pd.DataFrame:
    """Extracts statistical and kinematic features from 3-axis accelerometer signals
    to match the number of temporal windows extracted for other modalities.
    
    Args:
        acc_signals: Dict containing 'ACC_X', 'ACC_Y', 'ACC_Z' pandas Series.
        num_windows: Target number of windows (matching EDA/EEG/PPG rows).
        
    Returns:
        pd.DataFrame of shape (num_windows, num_acc_features).
    """
    x = acc_signals['ACC_X'].to_numpy(dtype=float)
    y = acc_signals['ACC_Y'].to_numpy(dtype=float)
    z = acc_signals['ACC_Z'].to_numpy(dtype=float)
    
    # Calculate magnitude: sqrt(x^2 + y^2 + z^2)
    mag = np.sqrt(x**2 + y**2 + z**2)
    
    n_samples = len(x)
    if num_windows <= 0 or n_samples == 0:
        return pd.DataFrame()
        
    window_size = n_samples // num_windows
    if window_size < 1:
        window_size = 1
        
    records = []
    for w in range(num_windows):
        start = w * window_size
        end = min((w + 1) * window_size, n_samples)
        if start >= n_samples:
            start = max(0, n_samples - window_size)
            end = n_samples
            
        x_win = x[start:end]
        y_win = y[start:end]
        z_win = z[start:end]
        m_win = mag[start:end]
        
        # Guard against empty windows
        if len(x_win) == 0:
            x_win = np.array([0.0])
            y_win = np.array([0.0])
            z_win = np.array([0.0])
            m_win = np.array([0.0])
            
        feat = {
            'ACC_X_mean': np.mean(x_win),
            'ACC_X_std': np.std(x_win),
            'ACC_X_min': np.min(x_win),
            'ACC_X_max': np.max(x_win),
            
            'ACC_Y_mean': np.mean(y_win),
            'ACC_Y_std': np.std(y_win),
            'ACC_Y_min': np.min(y_win),
            'ACC_Y_max': np.max(y_win),
            
            'ACC_Z_mean': np.mean(z_win),
            'ACC_Z_std': np.std(z_win),
            'ACC_Z_min': np.min(z_win),
            'ACC_Z_max': np.max(z_win),
            
            'ACC_Mag_mean': np.mean(m_win),
            'ACC_Mag_std': np.std(m_win),
            'ACC_Mag_max': np.max(m_win),
            'ACC_Mag_rms': np.sqrt(np.mean(m_win**2)),
            'ACC_movement_intensity': np.mean(np.abs(m_win - np.mean(m_win)))
        }
        records.append(feat)
        
    return pd.DataFrame(records)


def standardize_feature_columns(df: pd.DataFrame, modality_prefix: str) -> pd.DataFrame:
    """Standardizes column names by mapping Greek bands to English and adding modality prefix."""
    greek_map = {
        '\u03b1': 'alpha',
        '\u03b2': 'beta',
        '\u03b3': 'gamma',
        '\u03b4': 'delta',
        '\u03b8': 'theta',
    }
    clean_cols = {}
    seen = set()
    for col in df.columns:
        c_str = str(col).strip()
        for g_char, g_name in greek_map.items():
            c_str = c_str.replace(g_char, g_name)
        c_clean = c_str.encode('ascii', errors='ignore').decode('ascii').strip()
        c_clean = c_clean.replace(' ', '_').replace('/', '_div_').replace('-', '_')
        if not c_clean:
            c_clean = f"feat_{hash(str(col)) % 10000}"
        if not c_clean.startswith(f"{modality_prefix}_"):
            c_clean = f"{modality_prefix}_{c_clean}"
        # Ensure absolute uniqueness
        base_name = c_clean
        counter = 1
        while c_clean in seen:
            c_clean = f"{base_name}_{counter}"
            counter += 1
        seen.add(c_clean)
        clean_cols[col] = c_clean
    return df.rename(columns=clean_cols)



def get_modality_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identifies and groups all feature columns by modality based on prefix.
    
    Returns:
        Dict mapping modality name to list of feature column names:
        'EEG', 'PPG', 'EDA', 'ACC', 'TEMP', 'MULTIMODAL'
    """
    modalities = {
        'EEG': [c for c in df.columns if c.startswith('EEG_')],
        'PPG': [c for c in df.columns if c.startswith('PPG_')],
        'EDA': [c for c in df.columns if c.startswith('EDA_')],
        'ACC': [c for c in df.columns if c.startswith('ACC_')],
        'TEMP': [c for c in df.columns if c.startswith('TEMP_')],
    }
    all_features = []
    for mod_cols in modalities.values():
        all_features.extend(mod_cols)
    modalities['MULTIMODAL'] = all_features
    return modalities
