"""Regression modeling module for cognitive stress prediction.
Implements Ridge Regression (primary), Linear Regression, Polynomial Regression,
and Bayesian Linear Regression baselines with participant-level evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .evaluation import evaluate_regression, format_comparison_table
from .preprocessing import METADATA_COLUMNS


from sklearn.base import BaseEstimator, TransformerMixin

class DynamicPCA(BaseEstimator, TransformerMixin):
    """Adapts PCA components to min(max_components, n_features) to prevent dimensionality errors."""
    def __init__(self, max_components: int = 5, random_state: int = 42):
        self.max_components = max_components
        self.random_state = random_state
        self.pca_ = None

    def fit(self, X, y=None):
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        n_comp = max(1, min(self.max_components, n_features))
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca_.fit(X, y)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def get_regression_models(random_state: int = 42) -> Dict[str, Any]:
    """Defines the regression models as specified in PPT slide 6:
    - Ridge: Primary model (regularized linear)
    - Linear: Simple baseline reference
    - Polynomial: Degree-2 polynomial with adaptive PCA dimensionality reduction
    - Bayesian Linear: BayesianRidge representing parameter uncertainty
    """
    models = {
        'Ridge Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=10.0, random_state=random_state))
        ]),
        'Linear Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Polynomial Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', DynamicPCA(max_components=5, random_state=random_state)),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', Ridge(alpha=100.0, random_state=random_state))
        ]),
        'Bayesian Linear': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', BayesianRidge())
        ])
    }
    return models



def train_and_evaluate_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Weighted Nasa Score',
    modality_name: str = 'Multimodal',
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    """Trains and evaluates all regression models on participant-split train and test sets.
    
    Returns:
        (results_table, predictions_dict, best_model_name)
    """
    train_clean = train_df.dropna(subset=[target_col])
    test_clean = test_df.dropna(subset=[target_col])
    
    X_train = train_clean[feature_cols].copy()
    y_train = train_clean[target_col].to_numpy(dtype=float)
    
    X_test = test_clean[feature_cols].copy()
    y_test = test_clean[target_col].to_numpy(dtype=float)

    
    models = get_regression_models(random_state=random_state)
    results = []
    predictions = {}
    
    for name, model in models.items():
        # Fit strictly on train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        metrics = evaluate_regression(y_test, y_pred, model_name=name, modality=modality_name)
        results.append(metrics)
        
    results_df = pd.DataFrame(results).sort_values(by='MSE', ascending=True)
    best_model_name = results_df.iloc[0]['Model']
    
    return results_df, predictions, best_model_name


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_name: str,
    figures_dir: str
):
    """Plots and saves actual vs predicted scatter plot with reference line and residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='#2563eb', ax=axes[0])
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    axes[0].set_title(f'Actual vs Predicted: {model_name}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'Actual {target_name}', fontsize=10)
    axes[0].set_ylabel(f'Predicted {target_name}', fontsize=10)
    axes[0].legend()
    
    # 2. Residual distribution
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, color='#0284c7', ax=axes[1], bins=20)
    axes[1].axvline(0, color='r', linestyle='--', lw=1.5)
    axes[1].set_title(f'Residuals Distribution (Mean Error: {np.mean(residuals):.2f})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residual (Actual - Predicted)', fontsize=10)
    axes[1].set_ylabel('Count', fontsize=10)
    
    out_path = os.path.join(figures_dir, 'regression_actual_vs_predicted.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
