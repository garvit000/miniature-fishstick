"""Main entrypoint for Intelligent Detection of Cognitive Stress PBL-I Project.
Executes the end-to-end multimodal machine learning pipeline:
1. Dataset loading and feature integration (EPIStress)
2. Exploratory Data Analysis (EDA) and visualization
3. Participant-level train/test splitting (zero data leakage)
4. Continuous cognitive stress regression (Ridge vs baselines)
5. Discrete cognitive stress classification (SVM RBF vs baselines)
6. Modality comparison experiment (EEG, PPG, EDA, ACC, TEMP vs Multimodal)
7. Evidence-based model and modality evaluation summary
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add local directory to path for modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data_loader import build_multimodal_dataset, load_representative_raw_signals
from src.preprocessing import clean_feature_matrix, create_participant_train_test_split, METADATA_COLUMNS
from src.feature_engineering import get_modality_column_groups
from src.eda import run_full_eda
from src.regression import train_and_evaluate_regression, plot_actual_vs_predicted
from src.classification import train_and_evaluate_classification, plot_confusion_matrix
from src.modality_comparison import (
    run_modality_comparison_regression,
    run_modality_comparison_classification,
    plot_modality_comparison
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Cognitive Stress Detection ML Pipeline")
    parser.add_argument("--data-dir", type=str, default=os.path.join(current_dir, "data"),
                        help="Path to dataset root folder containing EPIStress directory")
    parser.add_argument("--output-dir", type=str, default=os.path.join(current_dir, "outputs"),
                        help="Path to outputs folder")
    parser.add_argument("--target-reg", type=str, default="Weighted Nasa Score",
                        help="Continuous target column for regression (e.g. 'Weighted Nasa Score', 'Mental Demand')")
    parser.add_argument("--target-clf", type=str, default="Stress Category",
                        help="Discrete target column for classification")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="Proportion of participants reserved for testing")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip generating EDA figures")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild feature matrix from raw files even if cache exists")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_state)

    figures_dir = os.path.join(args.output_dir, "figures")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cache_file = os.path.join(args.data_dir, "features_multimodal.csv")
    if args.force_rebuild and os.path.exists(cache_file):
        os.remove(cache_file)

    print("\n" + "="*70)
    print("   INTELLIGENT DETECTION OF COGNITIVE STRESS - PBL-I PIPELINE")
    print("   Multimodal Physiological Signals (EEG, PPG, EDA, ACC, TEMP)")
    print("="*70 + "\n")

    # 1. Load / Build Dataset
    print("[Step 1/6] Loading and structuring EPIStress multimodal dataset...")
    df = build_multimodal_dataset(args.data_dir, cache_file=cache_file)
    print(f"Loaded feature matrix: {df.shape[0]} temporal windows, {df.shape[1]} total columns.")
    
    # Identify feature columns vs metadata
    meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    modality_groups = get_modality_column_groups(df)
    
    print("\nModality feature distribution:")
    for mod in ['EEG', 'PPG', 'EDA', 'ACC', 'TEMP']:
        print(f"  • {mod:<12}: {len(modality_groups.get(mod, []))} features")
    print(f"  • Multimodal Total : {len(modality_groups.get('MULTIMODAL', []))} features\n")

    # 2. Exploratory Data Analysis
    if not args.skip_eda:
        print("[Step 2/6] Executing Exploratory Data Analysis (EDA)...")
        raw_signals = load_representative_raw_signals(args.data_dir, participant='ES140')
        run_full_eda(df, figures_dir, raw_signals=raw_signals, target_col=args.target_reg)
    else:
        print("[Step 2/6] Skipping EDA visualizations as requested.")

    # 3. Clean & Participant-level Split (Zero Leakage)
    print("[Step 3/6] Cleaning features and partitioning data by participant...")
    cleaned_df = clean_feature_matrix(df)
    train_df, test_df = create_participant_train_test_split(
        cleaned_df,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Update feature columns after cleaning
    feature_cols = [c for c in cleaned_df.columns if c not in meta_cols]

    # 4. Continuous Regression Benchmark
    print(f"[Step 4/6] Training regression models to predict continuous target: '{args.target_reg}'...")
    reg_results, reg_preds, best_reg_model = train_and_evaluate_regression(
        train_df,
        test_df,
        feature_cols=feature_cols,
        target_col=args.target_reg,
        modality_name='Multimodal',
        random_state=args.random_state
    )
    
    # Save regression comparison
    reg_csv_path = os.path.join(results_dir, "regression_model_comparison.csv")
    reg_results.to_csv(reg_csv_path, index=False)
    print(f"\nRegression Model Comparison Table saved to {reg_csv_path}:")
    print(reg_results.to_string(index=False))
    
    # Plot Actual vs Predicted for best regression model
    plot_actual_vs_predicted(
        y_true=test_df[args.target_reg].to_numpy(dtype=float),
        y_pred=reg_preds[best_reg_model],
        model_name=best_reg_model,
        target_name=args.target_reg,
        figures_dir=figures_dir
    )

    # 5. Discrete Classification Benchmark
    print(f"\n[Step 5/6] Training classification models to predict discrete target: '{args.target_clf}'...")
    clf_results, clf_preds, clf_cms, best_clf_model = train_and_evaluate_classification(
        train_df,
        test_df,
        feature_cols=feature_cols,
        target_col=args.target_clf,
        modality_name='Multimodal',
        random_state=args.random_state
    )
    
    # Save classification comparison
    clf_csv_path = os.path.join(results_dir, "classification_model_comparison.csv")
    clf_results.to_csv(clf_csv_path, index=False)
    print(f"\nClassification Model Comparison Table saved to {clf_csv_path}:")
    print(clf_results.to_string(index=False))
    
    # Plot Confusion Matrix for best classification model
    plot_confusion_matrix(
        cm=clf_cms[best_clf_model],
        classes=sorted(train_df[args.target_clf].unique()),
        model_name=best_clf_model,
        figures_dir=figures_dir
    )

    # 6. Multimodal vs Individual Modalities Experiment
    print("\n[Step 6/6] Executing Multimodal Experiment across individual modalities and fusion...")
    
    # Regression across modalities
    mod_reg_results = run_modality_comparison_regression(
        train_df,
        test_df,
        target_col=args.target_reg,
        preferred_model=best_reg_model,
        random_state=args.random_state
    )
    mod_reg_csv = os.path.join(results_dir, "modality_comparison_regression.csv")
    mod_reg_results.to_csv(mod_reg_csv, index=False)
    print(f"\nModality Comparison (Regression - {best_reg_model}):")
    print(mod_reg_results.to_string(index=False))

    # Classification across modalities
    mod_clf_results = run_modality_comparison_classification(
        train_df,
        test_df,
        target_col=args.target_clf,
        preferred_model=best_clf_model,
        random_state=args.random_state
    )
    mod_clf_csv = os.path.join(results_dir, "modality_comparison_classification.csv")
    mod_clf_results.to_csv(mod_clf_csv, index=False)
    print(f"\nModality Comparison (Classification - {best_clf_model}):")
    print(mod_clf_results.to_string(index=False))

    # Plot modality comparison charts
    plot_modality_comparison(mod_reg_results, mod_clf_results, figures_dir)

    # Determine best individual modality
    single_mods_clf = mod_clf_results[mod_clf_results['Modality'] != 'MULTIMODAL']
    best_single_clf = single_mods_clf.sort_values(by='F1-score', ascending=False).iloc[0]
    multimodal_clf = mod_clf_results[mod_clf_results['Modality'] == 'MULTIMODAL'].iloc[0]

    single_mods_reg = mod_reg_results[mod_reg_results['Modality'] != 'MULTIMODAL']
    best_single_reg = single_mods_reg.sort_values(by='MSE', ascending=True).iloc[0]
    multimodal_reg = mod_reg_results[mod_reg_results['Modality'] == 'MULTIMODAL'].iloc[0]

    clf_improvement = multimodal_clf['F1-score'] - best_single_clf['F1-score']
    reg_improvement = best_single_reg['MSE'] - multimodal_reg['MSE']

    print("\n" + "="*70)
    print("                    FINAL EXPERIMENTAL SUMMARY")
    print("="*70)
    print(f"1. Best Regression Model:       {best_reg_model} (MSE: {reg_results.iloc[0]['MSE']:.4f}, R²: {reg_results.iloc[0]['R²']:.4f})")
    print(f"2. Best Classification Model:   {best_clf_model} (Accuracy: {clf_results.iloc[0]['Accuracy']*100:.2f}%, F1: {clf_results.iloc[0]['F1-score']:.4f})")
    print(f"3. Best Individual Modality:    Classification -> {best_single_clf['Modality']} (F1: {best_single_clf['F1-score']:.4f}), Regression -> {best_single_reg['Modality']} (MSE: {best_single_reg['MSE']:.4f})")
    print(f"4. Multimodal Fusion Impact:    Classification F1: {multimodal_clf['F1-score']:.4f} vs Best Single ({best_single_clf['Modality']}): {best_single_clf['F1-score']:.4f}")
    if clf_improvement > 0:
        print(f"   --> Multimodal fusion IMPROVED classification F1 by +{clf_improvement:.4f} (+{clf_improvement/max(best_single_clf['F1-score'], 1e-4)*100:.2f}%)")
    elif clf_improvement == 0:
        print("   --> Multimodal fusion matched the top individual modality.")
    else:
        print(f"   --> Best single modality ({best_single_clf['Modality']}) outperformed multimodal fusion by {-clf_improvement:.4f}")

    print(f"   Regression MSE: {multimodal_reg['MSE']:.4f} vs Best Single ({best_single_reg['Modality']}): {best_single_reg['MSE']:.4f}")
    if reg_improvement > 0:
        print(f"   --> Multimodal fusion REDUCED regression MSE error by {reg_improvement:.4f} ({reg_improvement/max(best_single_reg['MSE'], 1e-4)*100:.2f}% lower error)")
    else:
        print(f"   --> Best single modality achieved lower MSE than multimodal fusion.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
