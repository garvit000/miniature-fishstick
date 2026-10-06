"""Preprocessing and data splitting module for cognitive stress machine learning.
Ensures participant-level train/test separation to eliminate data leakage across temporal windows,
and builds reproducible scikit-learn Pipelines with median imputation and standardization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


METADATA_COLUMNS = [
    'Participant',
    'Task',
    'Weighted Nasa Score',
    'Mental stress level',
    'Mental Demand',
    'Stress Category'
]


def clean_feature_matrix(df: pd.DataFrame, max_nan_ratio: float = 0.5) -> pd.DataFrame:
    """Cleans numeric feature columns by handling infinities, high-missing columns,
    and constant features while preserving metadata columns.
    """
    cleaned = df.copy()
    meta_cols = [c for c in METADATA_COLUMNS if c in cleaned.columns]
    feat_cols = [c for c in cleaned.columns if c not in meta_cols]
    
    # Replace infinite values with NaN safely
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    
    # Identify and drop features with excessive missing values
    missing_ratios = cleaned[feat_cols].isna().mean()
    drop_high_nan = missing_ratios[missing_ratios > max_nan_ratio].index.tolist()
    if drop_high_nan:
        print(f"Dropping {len(drop_high_nan)} features with > {max_nan_ratio*100:.0f}% missing values.")
        cleaned = cleaned.drop(columns=drop_high_nan)
        feat_cols = [c for c in feat_cols if c not in drop_high_nan]
        
    # Identify and drop constant / zero-variance features
    stds = cleaned[feat_cols].std(numeric_only=True)
    zero_var = stds[stds == 0].index.tolist()
    if zero_var:
        print(f"Dropping {len(zero_var)} constant features with zero variance.")
        cleaned = cleaned.drop(columns=zero_var)
        
    return cleaned



def create_participant_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset strictly by participant ID to ensure zero data leakage.
    Windows from the same participant are exclusively in train OR test, never both.
    
    Args:
        df: Input DataFrame containing 'Participant' column.
        test_size: Approximate fraction of participants reserved for testing.
        random_state: Seed for reproducible participant sampling.
        
    Returns:
        (train_df, test_df)
    """
    if 'Participant' not in df.columns:
        raise ValueError("DataFrame must contain 'Participant' column for participant-level splitting.")
        
    unique_participants = sorted(df['Participant'].unique())
    n_total = len(unique_participants)
    
    if n_total < 2:
        raise ValueError(f"At least 2 participants required for participant splitting, found {n_total}.")
        
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['Participant']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    train_parts = sorted(train_df['Participant'].unique())
    test_parts = sorted(test_df['Participant'].unique())
    
    # Assert zero overlap
    overlap = set(train_parts).intersection(set(test_parts))
    assert len(overlap) == 0, f"Critical: Data leakage detected! Overlapping participants: {overlap}"
    
    print("\n" + "="*50)
    print("PARTICIPANT-LEVEL SPLITTING STRATEGY (LEAKAGE PREVENTION)")
    print("="*50)
    print(f"Total participants: {n_total}")
    print(f"Training participants ({len(train_parts)}): {train_parts}")
    print(f"Testing participants  ({len(test_parts)}): {test_parts}")
    print(f"Train samples (windows): {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test samples (windows):  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print("Cross-contamination check: 0 participants overlapping between Train and Test.")
    print("="*50 + "\n")
    
    return train_df, test_df


def create_ml_pipeline(
    model: Any,
    feature_names: Optional[List[str]] = None,
    with_scaler: bool = True
) -> Pipeline:
    """Builds a scikit-learn Pipeline with median imputation and standard scaling.
    Preprocessing is fitted exclusively on the training subset.
    """
    steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]
    if with_scaler:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    
    return Pipeline(steps)


def get_participant_cv(
    df: pd.DataFrame,
    n_splits: int = 5
) -> GroupKFold:
    """Returns a GroupKFold cross-validator grouped by Participant."""
    n_participants = df['Participant'].nunique()
    effective_splits = min(n_splits, n_participants)
    return GroupKFold(n_splits=effective_splits)
