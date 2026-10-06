"""Classification modeling module for cognitive stress prediction.
Implements Support Vector Machine with RBF kernel (primary), Logistic Regression (baseline),
Linear Discriminant Analysis, and Gaussian Naive Bayes with participant-level evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

from .evaluation import evaluate_classification, format_comparison_table
from .preprocessing import METADATA_COLUMNS


def get_classification_models(random_state: int = 42) -> Dict[str, Any]:
    """Defines classification models as specified in PPT slide 7:
    - SVM (RBF Kernel): Primary model for high-dimensional, nonlinear boundaries
    - Logistic Regression: Probability-based baseline
    - Linear Discriminant Analysis: Discriminant function model
    - Gaussian Naive Bayes: Probabilistic model baseline
    """
    models = {
        'SVM (RBF Kernel)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=random_state))
        ]),
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=random_state))
        ]),
        'Linear Discriminant Analysis': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LinearDiscriminantAnalysis())
        ]),
        'Gaussian Naive Bayes': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ])
    }
    return models


def train_and_evaluate_classification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Stress Category',
    modality_name: str = 'Multimodal',
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
    """Trains and evaluates all classification models on participant-split train and test sets.
    
    Returns:
        (results_table, predictions_dict, confusion_matrices_dict, best_model_name)
    """
    train_clean = train_df.dropna(subset=[target_col])
    test_clean = test_df.dropna(subset=[target_col])
    
    X_train = train_clean[feature_cols].copy()
    y_train = train_clean[target_col].to_numpy()
    
    X_test = test_clean[feature_cols].copy()
    y_test = test_clean[target_col].to_numpy()

    
    models = get_classification_models(random_state=random_state)
    results = []
    predictions = {}
    cms = {}
    
    for name, model in models.items():
        # Fit strictly on train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        metrics = evaluate_classification(y_test, y_pred, model_name=name, modality=modality_name)
        cms[name] = metrics['Confusion_Matrix']
        results.append(metrics)
        
    results_df = format_comparison_table(results).sort_values(by='F1-score', ascending=False)
    best_model_name = results_df.iloc[0]['Model']
    
    return results_df, predictions, cms, best_model_name


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    model_name: str,
    figures_dir: str
):
    """Plots and saves normalized and counts confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=axes[0], annot_kws={'size': 13})
    axes[0].set_title(f'Confusion Matrix (Counts): {model_name}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)
    
    # 2. Normalized percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', vmin=0, vmax=1,
                xticklabels=classes, yticklabels=classes, ax=axes[1], annot_kws={'size': 13})
    axes[1].set_title(f'Confusion Matrix (Normalized Recall): {model_name}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=10)
    axes[1].set_ylabel('True Label', fontsize=10)
    
    out_path = os.path.join(figures_dir, 'classification_confusion_matrix.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
