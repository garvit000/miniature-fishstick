"""Modality comparison experiment module.
Evaluates machine learning models across individual modalities (EEG, PPG, EDA, Accelerometer, Temperature)
and all modalities combined (Multimodal) to determine whether fusion improves cognitive-stress detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

from .regression import get_regression_models
from .classification import get_classification_models
from .evaluation import evaluate_regression, evaluate_classification, format_comparison_table
from .feature_engineering import get_modality_column_groups


def run_modality_comparison_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'Weighted Nasa Score',
    preferred_model: str = 'Ridge Regression',
    random_state: int = 42
) -> pd.DataFrame:
    """Evaluates cognitive stress regression across each individual modality and multimodal fusion."""
    train_clean = train_df.dropna(subset=[target_col])
    test_clean = test_df.dropna(subset=[target_col])

    modality_groups = get_modality_column_groups(train_clean)
    modality_order = ['EEG', 'PPG', 'EDA', 'ACC', 'TEMP', 'MULTIMODAL']
    
    y_train = train_clean[target_col].to_numpy(dtype=float)
    y_test = test_clean[target_col].to_numpy(dtype=float)
    
    results = []
    
    for mod_name in modality_order:
        cols = modality_groups.get(mod_name, [])
        valid_cols = [c for c in cols if c in train_clean.columns and c in test_clean.columns]
        if not valid_cols:
            print(f"Warning: No valid columns found for modality {mod_name}")
            continue
            
        X_train = train_clean[valid_cols]
        X_test = test_clean[valid_cols]
        
        # Test models for this modality
        models = get_regression_models(random_state=random_state)
        # Select target model
        model = models.get(preferred_model, list(models.values())[0])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_regression(y_test, y_pred, model_name=preferred_model, modality=mod_name)
        metrics['Num_Features'] = len(valid_cols)
        results.append(metrics)
        
    df_results = pd.DataFrame(results)
    return df_results


def run_modality_comparison_classification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'Stress Category',
    preferred_model: str = 'SVM (RBF Kernel)',
    random_state: int = 42
) -> pd.DataFrame:
    """Evaluates cognitive stress classification across each individual modality and multimodal fusion."""
    train_clean = train_df.dropna(subset=[target_col])
    test_clean = test_df.dropna(subset=[target_col])

    modality_groups = get_modality_column_groups(train_clean)
    modality_order = ['EEG', 'PPG', 'EDA', 'ACC', 'TEMP', 'MULTIMODAL']
    
    y_train = train_clean[target_col].to_numpy()
    y_test = test_clean[target_col].to_numpy()
    
    results = []
    
    for mod_name in modality_order:
        cols = modality_groups.get(mod_name, [])
        valid_cols = [c for c in cols if c in train_clean.columns and c in test_clean.columns]
        if not valid_cols:
            print(f"Warning: No valid columns found for modality {mod_name}")
            continue
            
        X_train = train_clean[valid_cols]
        X_test = test_clean[valid_cols]

        
        models = get_classification_models(random_state=random_state)
        model = models.get(preferred_model, list(models.values())[0])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_classification(y_test, y_pred, model_name=preferred_model, modality=mod_name)
        metrics_dict = {k: v for k, v in metrics.items() if k != 'Confusion_Matrix'}
        metrics_dict['Num_Features'] = len(valid_cols)
        results.append(metrics_dict)
        
    df_results = pd.DataFrame(results)
    return df_results


def plot_modality_comparison(
    reg_comparison: pd.DataFrame,
    clf_comparison: pd.DataFrame,
    figures_dir: str
):
    """Plots comparative bar charts showing individual modalities vs multimodal performance."""
    # 1. Regression comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE
    sns.barplot(data=reg_comparison, x='Modality', y='MSE', palette='crest', ax=axes[0])
    axes[0].set_title('Regression MSE across Modalities (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Modality', fontsize=10)
    axes[0].set_ylabel('Mean Squared Error', fontsize=10)
    for p in axes[0].patches:
        h = p.get_height()
        if not np.isnan(h):
            axes[0].annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')
            
    # R2
    sns.barplot(data=reg_comparison, x='Modality', y='R²', palette='viridis', ax=axes[1])
    axes[1].set_title('Regression R² across Modalities (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Modality', fontsize=10)
    axes[1].set_ylabel('R² Score', fontsize=10)
    for p in axes[1].patches:
        h = p.get_height()
        if not np.isnan(h):
            axes[1].annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom' if h >= 0 else 'top', fontsize=9, xytext=(0, 2), textcoords='offset points')
            
    out_reg = os.path.join(figures_dir, 'modality_comparison_regression.png')
    plt.savefig(out_reg, dpi=300)
    plt.close()
    print(f"Saved: {out_reg}")
    
    # 2. Classification comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    sns.barplot(data=clf_comparison, x='Modality', y='Accuracy', palette='mako', ax=axes[0])
    axes[0].set_title('Classification Accuracy across Modalities (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Modality', fontsize=10)
    axes[0].set_ylabel('Accuracy', fontsize=10)
    axes[0].set_ylim(0, 1.05)
    for p in axes[0].patches:
        h = p.get_height()
        if not np.isnan(h):
            axes[0].annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')
            
    # F1-score
    sns.barplot(data=clf_comparison, x='Modality', y='F1-score', palette='flare', ax=axes[1])
    axes[1].set_title('Classification F1-score across Modalities (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Modality', fontsize=10)
    axes[1].set_ylabel('F1-score', fontsize=10)
    axes[1].set_ylim(0, 1.05)
    for p in axes[1].patches:
        h = p.get_height()
        if not np.isnan(h):
            axes[1].annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')
            
    out_clf = os.path.join(figures_dir, 'modality_comparison_classification.png')
    plt.savefig(out_clf, dpi=300)
    plt.close()
    print(f"Saved: {out_clf}")
