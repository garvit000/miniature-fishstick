"""Standardized evaluation metrics for regression and classification tasks.
Computes MSE, MAE, R2 for regression and Accuracy, Precision, Recall, F1 for classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    modality: str = "Multimodal"
) -> Dict[str, Any]:
    """Computes standard regression evaluation metrics (MSE, MAE, R²)."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    
    mse = float(mean_squared_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    
    return {
        'Model': model_name,
        'Modality': modality,
        'MSE': round(mse, 4),
        'MAE': round(mae, 4),
        'R²': round(r2, 4)
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    modality: str = "Multimodal"
) -> Dict[str, Any]:
    """Computes standard classification evaluation metrics (Accuracy, Precision, Recall, F1)."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    
    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, average='weighted', zero_division=0))
    rec = float(recall_score(y_t, y_p, average='weighted', zero_division=0))
    f1 = float(f1_score(y_t, y_p, average='weighted', zero_division=0))
    cm = confusion_matrix(y_t, y_p)
    
    return {
        'Model': model_name,
        'Modality': modality,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-score': round(f1, 4),
        'Confusion_Matrix': cm
    }


def format_comparison_table(results_list: list) -> pd.DataFrame:
    """Formats a list of evaluation dictionaries into a clean summary DataFrame."""
    clean_records = []
    for r in results_list:
        rec = {k: v for k, v in r.items() if k != 'Confusion_Matrix'}
        clean_records.append(rec)
    return pd.DataFrame(clean_records)
