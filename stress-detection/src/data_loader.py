"""Data loader for the EPIStress dataset.
Handles backward compatibility with pandas pickle formats, loads multimodal feature sets,
and structures targets for cognitive stress regression and classification.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Ensure compatibility between pandas 1.x pickle serializations and pandas 2.x
try:
    import pandas.core.internals.blocks as _blocks
    from pandas._libs.internals import BlockPlacement
    _orig_new_block = _blocks.new_block

    def _patched_new_block(values, placement, ndim=None, refs=None):
        if isinstance(placement, slice):
            placement = BlockPlacement(placement)
        return _orig_new_block(values, placement=placement, ndim=ndim, refs=refs)

    _blocks.new_block = _patched_new_block

    import pandas.core.indexes.base
    pandas.core.indexes.base.Int64Index = pd.Index
    pandas.core.indexes.base.Float64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
except Exception:
    pass


def safe_load_pickle(filepath: str) -> Any:
    """Safely unpickles pandas DataFrames/Series stored in legacy formats."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_available_participants(data_dir: str) -> List[str]:
    """Scans data_dir/EPIStress for participant directories."""
    epi_root = os.path.join(data_dir, 'EPIStress')
    if not os.path.exists(epi_root):
        epi_root = data_dir  # Fallback if points directly to EPIStress

    if not os.path.isdir(epi_root):
        return []

    dirs = [d for d in os.listdir(epi_root) if os.path.isdir(os.path.join(epi_root, d)) and d.startswith('ES')]
    return sorted(dirs)


def load_task_labels(participant_id: str, data_dir: str) -> Optional[pd.DataFrame]:
    """Loads Lab Task_Labels.csv for a given participant."""
    possible_paths = [
        os.path.join(data_dir, 'EPIStress', participant_id, 'Lab', 'Task_Labels.csv'),
        os.path.join(data_dir, participant_id, 'Lab', 'Task_Labels.csv'),
    ]
    for p in possible_paths:
        if os.path.isfile(p):
            try:
                df = pd.read_csv(p)
                df['Participant'] = participant_id
                return df
            except Exception as e:
                print(f"Warning: Could not read labels from {p}: {e}")
    return None


def load_task_features(
    participant_id: str,
    task: str,
    modality: str,
    data_dir: str,
    window_sec: int = 30
) -> Optional[pd.DataFrame]:
    """Loads pre-extracted feature pickle for EDA, EEG, PPG, or TEMP."""
    filename = f"{modality}_features_{window_sec}_sec.pickle"
    possible_paths = [
        os.path.join(data_dir, 'EPIStress', participant_id, 'Lab', 'Features', task, filename),
        os.path.join(data_dir, participant_id, 'Lab', 'Features', task, filename),
    ]
    for p in possible_paths:
        if os.path.isfile(p):
            try:
                data = safe_load_pickle(p)
                if isinstance(data, pd.DataFrame):
                    return data
                elif isinstance(data, pd.Series):
                    return data.to_frame()
            except Exception as e:
                pass
    return None


def load_acc_signals(
    participant_id: str,
    task: str,
    data_dir: str
) -> Optional[Dict[str, pd.Series]]:
    """Loads segmented ACC signals (X, Y, Z) for the given task."""
    acc_data = {}
    for axis in ['ACC_X', 'ACC_Y', 'ACC_Z']:
        filename = f"E4__{axis}.pickle"
        possible_paths = [
            os.path.join(data_dir, 'EPIStress', participant_id, 'Lab', 'Labeled', task, filename),
            os.path.join(data_dir, participant_id, 'Lab', 'Labeled', task, filename),
        ]
        found = False
        for p in possible_paths:
            if os.path.isfile(p):
                try:
                    s = safe_load_pickle(p)
                    if isinstance(s, pd.Series):
                        acc_data[axis] = s
                        found = True
                        break
                except Exception:
                    pass
        if not found:
            return None
    return acc_data


def load_representative_raw_signals(data_dir: str, participant: str = 'ES140') -> Dict[str, Dict[str, pd.Series]]:
    """Loads representative raw signal traces for EDA baseline vs stress visualization.
    Returns: {
        'baseline': {'EEG': Series, 'PPG': Series, 'EDA': Series},
        'stress': {'EEG': Series, 'PPG': Series, 'EDA': Series}
    }
    """
    results = {'baseline': {}, 'stress': {}}
    task_map = {
        'baseline': 'relaxation_video',
        'stress': 'arithmetix_hard'
    }
    signal_map = {
        'EEG': 'Muse__RAW_AF7.pickle',
        'PPG': 'E4__BVP.pickle',
        'EDA': 'E4__EDA.pickle'
    }

    for state, task in task_map.items():
        for sig_name, filename in signal_map.items():
            possible_paths = [
                os.path.join(data_dir, 'EPIStress', participant, 'Lab', 'Labeled', task, filename),
                os.path.join(data_dir, participant, 'Lab', 'Labeled', task, filename),
            ]
            for p in possible_paths:
                if os.path.isfile(p):
                    try:
                        s = safe_load_pickle(p)
                        results[state][sig_name] = s
                        break
                    except Exception:
                        pass
    return results


def build_multimodal_dataset(
    data_dir: str,
    cache_file: Optional[str] = None,
    window_sec: int = 30
) -> pd.DataFrame:
    """Aggregates all available participant data into a unified multimodal dataset.
    Aligns temporal windows across EEG, PPG, EDA, Accelerometer, and Temperature,
    and assigns target cognitive stress scores from Task_Labels.
    """
    if cache_file and os.path.isfile(cache_file) and os.path.getsize(cache_file) > 1024:
        print(f"Loading multimodal dataset from existing cache: {cache_file}")
        return pd.read_csv(cache_file)

    from .feature_engineering import extract_accelerometer_features, standardize_feature_columns

    participants = get_available_participants(data_dir)
    print(f"Discovered {len(participants)} participants in {data_dir}: {participants}")

    all_task_dfs = []

    for p in participants:
        labels_df = load_task_labels(p, data_dir)
        if labels_df is None or labels_df.empty:
            continue

        for _, row in labels_df.iterrows():
            task = str(row['Task']).strip()
            
            # Load pre-extracted features
            eda_df = load_task_features(p, task, 'EDA', data_dir, window_sec=window_sec)
            eeg_df = load_task_features(p, task, 'EEG', data_dir, window_sec=window_sec)
            ppg_df = load_task_features(p, task, 'PPG', data_dir, window_sec=window_sec)
            temp_df = load_task_features(p, task, 'TEMP', data_dir, window_sec=window_sec)

            if eda_df is None or eeg_df is None or ppg_df is None or temp_df is None:
                continue

            # Determine aligned window length
            min_w = min(len(eda_df), len(eeg_df), len(ppg_df), len(temp_df))
            if min_w < 1:
                continue

            # Load and engineer ACC features
            acc_signals = load_acc_signals(p, task, data_dir)
            if acc_signals is not None:
                acc_df = extract_accelerometer_features(acc_signals, num_windows=min_w)
            else:
                # Fallback if raw ACC signals are missing for this specific task
                acc_df = pd.DataFrame(index=range(min_w))

            # Standardize column prefixes
            eda_df_clean = standardize_feature_columns(eda_df.iloc[:min_w].reset_index(drop=True), 'EDA')
            eeg_df_clean = standardize_feature_columns(eeg_df.iloc[:min_w].reset_index(drop=True), 'EEG')
            ppg_df_clean = standardize_feature_columns(ppg_df.iloc[:min_w].reset_index(drop=True), 'PPG')
            temp_df_clean = standardize_feature_columns(temp_df.iloc[:min_w].reset_index(drop=True), 'TEMP')
            acc_df_clean = standardize_feature_columns(acc_df.iloc[:min_w].reset_index(drop=True), 'ACC') if not acc_df.empty else pd.DataFrame(index=range(min_w))

            task_multimodal = pd.concat([
                eda_df_clean,
                eeg_df_clean,
                ppg_df_clean,
                temp_df_clean,
                acc_df_clean
            ], axis=1)

            # Assign targets and metadata
            task_multimodal['Participant'] = p
            task_multimodal['Task'] = task
            
            # Target columns
            weighted_nasa = row.get('Weighted Nasa Score', np.nan)
            mental_stress = row.get('Mental stress level', np.nan)
            mental_demand = row.get('Mental Demand', np.nan)

            task_multimodal['Weighted Nasa Score'] = float(weighted_nasa) if pd.notna(weighted_nasa) else np.nan
            task_multimodal['Mental stress level'] = float(mental_stress) if pd.notna(mental_stress) else np.nan
            task_multimodal['Mental Demand'] = float(mental_demand) if pd.notna(mental_demand) else np.nan
            
            # Discrete Stress Category: Low vs High
            # In EPIStress Likert 1-5, <= 3 represents low/mild baseline or mild difficulty, > 3 represents high stress
            if pd.notna(mental_stress):
                task_multimodal['Stress Category'] = 'High' if mental_stress > 3.0 else 'Low'
            elif pd.notna(weighted_nasa):
                task_multimodal['Stress Category'] = 'High' if weighted_nasa >= 50.0 else 'Low'
            else:
                task_multimodal['Stress Category'] = 'Low'

            all_task_dfs.append(task_multimodal)

    if not all_task_dfs:
        raise RuntimeError("No matching multimodal task data could be loaded. Verify dataset files in data directory.")

    combined_df = pd.concat(all_task_dfs, ignore_index=True)
    print(f"Successfully constructed feature matrix with {combined_df.shape[0]} samples and {combined_df.shape[1]} columns.")

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        combined_df.to_csv(cache_file, index=False)
        print(f"Saved feature matrix cache to {cache_file}")

    return combined_df

