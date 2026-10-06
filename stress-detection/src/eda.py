"""Exploratory Data Analysis (EDA) module for cognitive stress analysis.
Generates comprehensive visual evidence including dataset structure, missing values,
stress distributions, feature correlations, and representative raw physiological waveforms.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any

# Set modern scientific aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'Arial',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.4,
    'figure.autolayout': True
})


def plot_dataset_overview(df: pd.DataFrame, figures_dir: str):
    """Visualizes participant distributions, task breakdown, and modality composition."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Samples per participant
    part_counts = df['Participant'].value_counts().sort_index()
    sns.barplot(x=part_counts.index, y=part_counts.values, ax=axes[0], color='#3b82f6')
    axes[0].set_title('Temporal Windows per Participant', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Participant ID', fontsize=10)
    axes[0].set_ylabel('Number of Windows', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Samples per cognitive task
    task_counts = df['Task'].value_counts()
    sns.barplot(y=task_counts.index, x=task_counts.values, ax=axes[1], color='#10b981')
    axes[1].set_title('Samples per Experimental Task', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sample Count', fontsize=10)
    axes[1].set_ylabel('Task', fontsize=10)
    
    # 3. Modality feature counts
    modality_counts = {
        'EEG': len([c for c in df.columns if c.startswith('EEG_')]),
        'PPG': len([c for c in df.columns if c.startswith('PPG_')]),
        'EDA': len([c for c in df.columns if c.startswith('EDA_')]),
        'ACC': len([c for c in df.columns if c.startswith('ACC_')]),
        'TEMP': len([c for c in df.columns if c.startswith('TEMP_')]),
    }
    axes[2].pie(
        list(modality_counts.values()),
        labels=[f"{k} ({v})" for k, v in modality_counts.items()],
        autopct='%1.1f%%',
        colors=['#6366f1', '#ec4899', '#f59e0b', '#06b6d4', '#84cc16'],
        startangle=140
    )
    axes[2].set_title('Feature Count Breakdown by Modality', fontsize=12, fontweight='bold')
    
    out_path = os.path.join(figures_dir, 'eda_dataset_overview.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_missing_values(df: pd.DataFrame, figures_dir: str):
    """Analyzes and plots missing value patterns across modalities."""
    meta_cols = ['Participant', 'Task', 'Weighted Nasa Score', 'Mental stress level', 'Mental Demand', 'Stress Category']
    feat_cols = [c for c in df.columns if c not in meta_cols]
    
    # Calculate missingness by modality
    mod_missing = {}
    for mod in ['EEG', 'PPG', 'EDA', 'ACC', 'TEMP']:
        m_cols = [c for c in feat_cols if c.startswith(f"{mod}_")]
        if m_cols:
            mod_missing[mod] = df[m_cols].isna().mean().mean() * 100
        else:
            mod_missing[mod] = 0.0
            
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(list(mod_missing.keys()), list(mod_missing.values()), color='#ef4444', alpha=0.85, width=0.5)
    ax.set_title('Missing Value Percentage by Physiological Modality', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Missingness (%)', fontsize=10)
    ax.set_ylim(0, max(max(mod_missing.values(), default=0) * 1.5, 5))
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.1, f"{h:.2f}%", ha='center', va='bottom', fontsize=9)
        
    out_path = os.path.join(figures_dir, 'eda_missing_values.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_target_distributions(df: pd.DataFrame, figures_dir: str):
    """Plots distributions of continuous cognitive-stress targets and discrete stress categories."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Weighted NASA-TLX continuous distribution
    if 'Weighted Nasa Score' in df.columns:
        sns.histplot(df['Weighted Nasa Score'].dropna(), kde=True, ax=axes[0], color='#8b5cf6', bins=15)
        axes[0].set_title('Weighted NASA-TLX Score Distribution (Continuous)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Weighted NASA Score (0-100)', fontsize=10)
        axes[0].set_ylabel('Sample Count', fontsize=10)
        
    # 2. Mental stress level (Likert 1-5)
    if 'Mental stress level' in df.columns:
        sns.histplot(df['Mental stress level'].dropna(), discrete=True, ax=axes[1], color='#f97316', shrink=0.8)
        axes[1].set_title('Self-Reported Mental Stress Level (1-5)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Likert Score', fontsize=10)
        axes[1].set_ylabel('Sample Count', fontsize=10)
        
    # 3. Discrete Stress Category (Binary / Multiclass)
    if 'Stress Category' in df.columns:
        cat_counts = df['Stress Category'].value_counts()
        sns.barplot(x=cat_counts.index.astype(str), y=cat_counts.values, ax=axes[2], palette=['#10b981', '#ef4444'])
        axes[2].set_title('Cognitive Stress Category Distribution (Discrete)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Stress Category', fontsize=10)
        axes[2].set_ylabel('Sample Count', fontsize=10)
        for i, val in enumerate(cat_counts.values):
            axes[2].text(i, val + max(cat_counts.values)*0.01, f"{val} ({val/len(df)*100:.1f}%)", ha='center', fontsize=9)
            
    out_path = os.path.join(figures_dir, 'eda_target_distributions.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, figures_dir: str, target_col: str = 'Weighted Nasa Score'):
    """Plots Pearson correlation heatmap between top representative modality features and the target."""
    meta_cols = ['Participant', 'Task', 'Weighted Nasa Score', 'Mental stress level', 'Mental Demand', 'Stress Category']
    feat_cols = [c for c in df.columns if c not in meta_cols]
    
    # Pick representative features per modality
    rep_features = []
    for mod in ['EDA', 'PPG', 'TEMP', 'ACC', 'EEG']:
        m_cols = [c for c in feat_cols if c.startswith(f"{mod}_")]
        if m_cols and target_col in df.columns:
            corrs = df[m_cols].apply(lambda x: x.corr(df[target_col])).abs()
            top_cols = corrs.nlargest(3).index.tolist()
            rep_features.extend(top_cols)
            
    if not rep_features or target_col not in df.columns:
        return
        
    selected_cols = rep_features + [target_col]
    corr_matrix = df[selected_cols].corr()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title(f'Pearson Correlation Matrix: Key Multimodal Features vs {target_col}', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    out_path = os.path.join(figures_dir, 'eda_correlation_matrix.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_representative_signals(raw_signals: Dict[str, Dict[str, pd.Series]], figures_dir: str):
    """Plots representative raw physiological signals: EEG (Muse AF7), PPG (E4 BVP), and EDA (E4 EDA)
    contrasting Relaxed Baseline vs Cognitive Stress.
    """
    if not raw_signals or 'baseline' not in raw_signals or 'stress' not in raw_signals:
        print("Notice: Representative raw signals not provided for plotting.")
        return
        
    baseline = raw_signals['baseline']
    stress = raw_signals['stress']
    
    if not baseline or not stress:
        return
        
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex='col')
    
    modalities = [
        ('EEG', 'EEG (Muse AF7) [μV]', '#3b82f6', 256),     # ~256 Hz
        ('PPG', 'PPG (Empatica BVP)', '#ec4899', 64),       # 64 Hz
        ('EDA', 'EDA (Empatica Conductance) [μS]', '#f59e0b', 4)  # 4 Hz
    ]
    
    for row_idx, (mod_key, label, color, fs) in enumerate(modalities):
        # 1. Baseline trace (5 seconds window)
        if mod_key in baseline and len(baseline[mod_key]) > 0:
            s_base = baseline[mod_key].values
            dur_samples = min(int(5 * fs), len(s_base))
            t_base = np.linspace(0, dur_samples / fs, dur_samples)
            axes[row_idx, 0].plot(t_base, s_base[:dur_samples], color=color, lw=1.2)
            axes[row_idx, 0].set_ylabel(label, fontsize=10, fontweight='bold')
            if row_idx == 0:
                axes[row_idx, 0].set_title('Relaxed Baseline (Relaxation Video)', fontsize=11, fontweight='bold')
            if row_idx == 2:
                axes[row_idx, 0].set_xlabel('Time (seconds)', fontsize=10)
                
        # 2. Stress trace (5 seconds window)
        if mod_key in stress and len(stress[mod_key]) > 0:
            s_stress = stress[mod_key].values
            dur_samples = min(int(5 * fs), len(s_stress))
            t_stress = np.linspace(0, dur_samples / fs, dur_samples)
            axes[row_idx, 1].plot(t_stress, s_stress[:dur_samples], color='#ef4444', lw=1.2)
            if row_idx == 0:
                axes[row_idx, 1].set_title('Cognitive Stress (Arithmetic Hard)', fontsize=11, fontweight='bold')
            if row_idx == 2:
                axes[row_idx, 1].set_xlabel('Time (seconds)', fontsize=10)
                
    out_path = os.path.join(figures_dir, 'eda_representative_signals.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def run_full_eda(
    df: pd.DataFrame,
    figures_dir: str,
    raw_signals: Optional[Dict[str, Dict[str, pd.Series]]] = None,
    target_col: str = 'Weighted Nasa Score'
):
    """Runs the complete EDA suite and outputs all figures."""
    os.makedirs(figures_dir, exist_ok=True)
    print("\n--- Running Exploratory Data Analysis (EDA) ---")
    plot_dataset_overview(df, figures_dir)
    plot_missing_values(df, figures_dir)
    plot_target_distributions(df, figures_dir)
    plot_correlation_heatmap(df, figures_dir, target_col=target_col)
    if raw_signals:
        plot_representative_signals(raw_signals, figures_dir)
    print("EDA execution completed successfully.\n")
