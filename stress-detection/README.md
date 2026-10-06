# Intelligent Detection of Cognitive Stress

**PBL-I Project: Multimodal Physiological and Behavioral Signal Processing for Cognitive Stress Detection**

---

## Overview

This project implements an end-to-end machine learning pipeline to investigate whether multimodal physiological signals improve the detection and quantification of cognitive stress compared to individual sensor modalities. 

The project uses the **EPIStress** dataset, capturing five complementary physiological and behavioral views:
1. **EEG (Electroencephalography)** — Neural activity via Muse S headband (frontal and temporal channels).
2. **PPG (Photoplethysmography)** — Pulse-related cardiovascular response (Heart Rate and Heart Rate Variability features: HR, HRV MeanNN, SDNN, RMSSD).
3. **EDA (Electrodermal Activity)** — Sympathetic autonomic arousal (Skin Conductance Response peaks, tonic component, phasic component).
4. **Accelerometer (ACC)** — 3-axis movement and kinematic context (mean, std, min, max, RMS, movement intensity).
5. **Temperature (TEMP)** — Peripheral autonomic vasomotor response (mean temperature, temperature standard deviation).

---

## Machine Learning Tasks

### 1. Continuous Cognitive Stress Regression
Estimates continuous cognitive stress and mental workload (e.g., Weighted NASA-TLX score, 0–100):
- **Primary Model**: Ridge Regression (L2 regularization to stabilize correlated multimodal features).
- **Baselines**:
  - Linear Regression (interpretable baseline reference).
  - Polynomial Regression (degree 2 with PCA dimensionality reduction to capture nonlinear interactions).
  - Bayesian Linear Regression (BayesianRidge representing parameter uncertainty).
- **Evaluation Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), Coefficient of Determination ($R^2$).

### 2. Discrete Cognitive Stress Classification
Classifies cognitive stress states into discrete categories (Low Stress vs. High Stress):
- **Primary Model**: Support Vector Machine with Radial Basis Function kernel (SVM RBF).
- **Baselines**:
  - Logistic Regression (probability-based linear baseline).
  - Linear Discriminant Analysis (LDA) (Fisher discriminant boundary).
  - Gaussian Naive Bayes (probabilistic reference).
- **Evaluation Metrics**: Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted), Confusion Matrix.

### 3. Multimodal Comparative Experiment
To answer the core research question (*"Does multimodal fusion outperform individual modalities?"*), identical pipelines are evaluated on:
1. **EEG only**
2. **PPG only**
3. **EDA only**
4. **Accelerometer only**
5. **Temperature only**
6. **Multimodal fusion (All modalities combined)**

---

## Key Methodological Principles

- **Zero Data Leakage**: Temporal windows belonging to the same participant are kept strictly together. The dataset is partitioned using **Participant-Level Splitting** (`GroupShuffleSplit` / `GroupKFold`), testing true generalization to unseen participants.
- **Scikit-Learn Pipelines**: All preprocessing steps (median imputation, standardization) are encapsulated within `Pipeline` objects and fitted exclusively on training participants.
- **Evidence-Based Model Selection**: Model and modality selections are determined solely by empirical evaluation metrics.

---

## Project Structure

```
stress-detection/
├── data/
│   ├── EPIStress/               # Participant directories (ES140, ES141, ...)
│   └── features_multimodal.csv  # Preprocessed consolidated feature matrix
├── src/
│   ├── data_loader.py           # EPIStress parsing and backward-compatible unpickler
│   ├── feature_engineering.py   # Accelerometer feature extraction & prefix tagging
│   ├── preprocessing.py         # Data cleaning, participant-level splitting & Pipelines
│   ├── eda.py                   # Exploratory data analysis & publication-ready plots
│   ├── regression.py            # Ridge, Linear, Polynomial, Bayesian regression models
│   ├── classification.py        # SVM RBF, Logistic Regression, LDA, GNB classifiers
│   ├── modality_comparison.py   # 5 modalities vs Multimodal fusion benchmark
│   └── evaluation.py            # Standardized regression and classification metrics
├── outputs/
│   ├── figures/                 # Generated EDA, actual-vs-predicted, and confusion matrices
│   └── results/                 # Evaluation metric comparison CSV files
├── main.py                      # Main pipeline execution script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Installation & Requirements

Ensure Python 3.9+ is installed. Install required packages:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Run from Workspace Root:
```bash
python main.py
```

### Run with Custom Options:
```bash
# Custom test split ratio (e.g. 20% test participants)
python stress-detection/main.py --test-size 0.20

# Force rebuild the feature matrix cache
python stress-detection/main.py --force-rebuild

# Skip generating EDA plots for faster evaluation
python stress-detection/main.py --skip-eda
```

---

## Outputs Generated

Upon completion, all outputs are saved in `outputs/`:
- **Figures** (`outputs/figures/`):
  - `eda_dataset_overview.png`: Sample counts, task breakdown, modality distributions.
  - `eda_missing_values.png`: Completeness analysis across sensor modalities.
  - `eda_target_distributions.png`: Continuous NASA-TLX and Likert distributions.
  - `eda_correlation_matrix.png`: Multimodal feature correlations with stress targets.
  - `eda_representative_signals.png`: Raw EEG, PPG, EDA traces comparing Baseline vs Stress.
  - `regression_actual_vs_predicted.png`: Scatter plot & residual distribution.
  - `classification_confusion_matrix.png`: Counts and normalized recall heatmaps.
  - `modality_comparison_regression.png`: MSE and $R^2$ across all modalities.
  - `modality_comparison_classification.png`: Accuracy and F1-score across all modalities.
- **Results Tables** (`outputs/results/`):
  - `regression_model_comparison.csv`
  - `classification_model_comparison.csv`
  - `modality_comparison_regression.csv`
  - `modality_comparison_classification.csv`
