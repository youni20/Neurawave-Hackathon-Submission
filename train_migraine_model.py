#!/usr/bin/env python3
"""
Main training script for migraine prediction model.
Orchestrates all modules to train a bulletproof XGBoost model.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from migraine_model.data_loader import load_data, validate_data, prepare_target
from migraine_model.feature_engineering import FeatureEngineer
from migraine_model.data_split import stratified_split
from migraine_model.train_xgboost import optimize_hyperparameters, train_final_model, get_feature_importance
from migraine_model.evaluate_model import evaluate_model, plot_roc_curve, plot_calibration_curve, plot_feature_importance
from migraine_model.save_model import save_model, save_evaluation_report


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train XGBoost model for migraine prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_migraine_model.py --input combined_data.csv --output models/
  
  # Train with custom hyperparameter trials
  python train_migraine_model.py --input combined_data.csv --output models/ --trials 200
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='combined_data.csv',
        help='Input CSV file path (default: combined_data.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for model artifacts (default: models)'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of hyperparameter optimization trials (default: 100)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--use-distribution-shift',
        action='store_true',
        help='Use separate train/val/test files with distribution shift (requires --train-file, --val-file, --test-file)'
    )
    
    parser.add_argument(
        '--train-file',
        type=str,
        default=None,
        help='Training data file (required if --use-distribution-shift)'
    )
    
    parser.add_argument(
        '--val-file',
        type=str,
        default=None,
        help='Validation data file (required if --use-distribution-shift)'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        default=None,
        help='Test data file (required if --use-distribution-shift)'
    )
    
    args = parser.parse_args()
    
    # Validate input file (only if not using distribution shift)
    if not args.use_distribution_shift:
        if not Path(args.input).exists():
            print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        # Validate distribution shift files
        for file_arg, file_path in [('--train-file', args.train_file), 
                                    ('--val-file', args.val_file), 
                                    ('--test-file', args.test_file)]:
            if not Path(file_path).exists():
                print(f"Error: {file_arg} file '{file_path}' not found.", file=sys.stderr)
                sys.exit(1)
    
    print("=" * 80)
    print("MIGRAINE PREDICTION MODEL TRAINING")
    print("=" * 80)
    if args.use_distribution_shift:
        print(f"Using distribution shift: YES")
        print(f"  Train file: {args.train_file}")
        print(f"  Val file:   {args.val_file}")
        print(f"  Test file:  {args.test_file}")
    else:
        print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Hyperparameter trials: {args.trials}")
    print(f"Random state: {args.random_state}")
    print("=" * 80)
    
    try:
        # Set random seeds
        np.random.seed(args.random_state)
        
        # 1. Load data (with or without distribution shift)
        if args.use_distribution_shift:
            if not args.train_file or not args.val_file or not args.test_file:
                print("Error: --use-distribution-shift requires --train-file, --val-file, and --test-file", file=sys.stderr)
                sys.exit(1)
            
            print("\n" + "=" * 80)
            print("LOADING DATA WITH DISTRIBUTION SHIFT")
            print("=" * 80)
            print("Using separate train/val/test files with different distributions")
            print("This helps detect overfitting by testing generalization to different data distributions.")
            print("=" * 80)
            
            # Load separate files
            df_train = load_data(args.train_file)
            df_val = load_data(args.val_file)
            df_test = load_data(args.test_file)
            
            # Validate each
            print("\nValidating training data...")
            validation_report_train = validate_data(df_train)
            print("\nValidating validation data...")
            validation_report_val = validate_data(df_val)
            print("\nValidating test data...")
            validation_report_test = validate_data(df_test)
            
            # Prepare targets
            X_train_raw, y_train = prepare_target(df_train)
            X_val_raw, y_val = prepare_target(df_val)
            X_test_raw, y_test = prepare_target(df_test)
            
            # Feature engineering (fit on train, transform all)
            print("\n" + "=" * 80)
            print("FEATURE ENGINEERING")
            print("=" * 80)
            feature_engineer = FeatureEngineer()
            X_train = feature_engineer.fit_transform(X_train_raw)
            X_val = feature_engineer.transform(X_val_raw)
            X_test = feature_engineer.transform(X_test_raw)
            
            print(f"\nData split (from separate files):")
            print(f"  Training:   {len(X_train):,} samples")
            print(f"  Validation: {len(X_val):,} samples")
            print(f"  Test:       {len(X_test):,} samples")
            
        else:
            # Original approach: single file, random split
            # 1. Load and validate data
            df = load_data(args.input)
            validation_report = validate_data(df)
            
            # 2. Prepare target
            X, y = prepare_target(df)
            
            # 3. Feature engineering
            print("\n" + "=" * 80)
            print("FEATURE ENGINEERING")
            print("=" * 80)
            feature_engineer = FeatureEngineer()
            X_processed = feature_engineer.fit_transform(X)
            
            # 4. Data splitting
            X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
                X_processed, y,
                test_size=0.15,
                val_size=0.15,
                random_state=args.random_state
            )
        
        # 5. Hyperparameter optimization
        best_params = optimize_hyperparameters(
            X_train, y_train,
            X_val, y_val,
            n_trials=args.trials,
            random_state=args.random_state
        )
        
        # 6. Train final model
        model = train_final_model(
            X_train, y_train,
            X_val, y_val,
            best_params,
            n_estimators=200  # Much reduced: 200 to prevent overfitting
        )
        
        # 7. Get feature importance
        feature_names = feature_engineer.get_feature_names()
        importance_df = get_feature_importance(model, feature_names)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Feature importance validation
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            top_feature_pct = (importance_df.iloc[0]['importance'] / total_importance) * 100
            top3_importance = importance_df.head(3)['importance'].sum()
            top3_pct = (top3_importance / total_importance) * 100
            num_features_used = (importance_df['importance'] > 0).sum()
            
            print(f"\nFeature Importance Analysis:")
            print(f"  Features with non-zero importance: {num_features_used}/{len(importance_df)}")
            print(f"  Top feature accounts for: {top_feature_pct:.2f}% of total importance")
            print(f"  Top 3 features account for: {top3_pct:.2f}% of total importance")
            
            # PHASE 3: Enhanced validation - check feature groups
            feature_groups = feature_engineer.get_feature_groups() if hasattr(feature_engineer, 'get_feature_groups') else {}
            if feature_groups:
                print(f"\nFeature Group Analysis:")
                for group_name, group_features in feature_groups.items():
                    if group_features:
                        group_importance = importance_df[importance_df['feature'].isin(group_features)]['importance'].sum()
                        group_pct = (group_importance / total_importance * 100) if total_importance > 0 else 0
                        print(f"  {group_name.capitalize()} group: {group_pct:.2f}% of total importance ({len(group_features)} features)")
            
            # PHASE 3: Check if predictor features are in top 10
            top10_features = importance_df.head(10)['feature'].tolist()
            predictor_features = feature_groups.get('predictor', [])
            predictor_in_top10 = [f for f in top10_features if f in predictor_features]
            if len(predictor_in_top10) >= 3:
                print(f"  ✓ Predictor features in top 10: {len(predictor_in_top10)}/{len(predictor_in_top10)} (good)")
            else:
                print(f"  ⚠ WARNING: Only {len(predictor_in_top10)} predictor features in top 10! Expected: ≥3")
            
            # PHASE 0: Check if symptom features dominate top 3
            symptom_features = feature_groups.get('symptom', [])
            top3_features = importance_df.head(3)['feature'].tolist()
            symptom_in_top3 = [f for f in top3_features if any(symptom in f.lower() for symptom in ['mood', 'step_count', 'screen_brightness'])]
            if symptom_in_top3:
                symptom_top3_importance = importance_df[importance_df['feature'].isin(symptom_in_top3)]['importance'].sum()
                symptom_top3_pct = (symptom_top3_importance / top3_importance * 100) if top3_importance > 0 else 0
                # AGGRESSIVE: Stricter threshold (was 50%, now 25%)
                if symptom_top3_pct > 25:
                    print(f"  ⚠ WARNING: Symptom features dominate top 3 ({symptom_top3_pct:.2f}% of top 3 importance)!")
                    print(f"    Symptom features in top 3: {symptom_in_top3}")
                else:
                    print(f"  ✓ Symptom features in top 3: {symptom_top3_pct:.2f}% (acceptable)")
            
            # AGGRESSIVE: Stricter thresholds to match training constraints
            if top_feature_pct > 20:  # AGGRESSIVE: Was 50%, now 20%
                print(f"  ⚠ WARNING: Single feature has >20% importance! Model may be overfitting.")
            if top3_pct > 35:  # AGGRESSIVE: Was 60%, now 35%
                print(f"  ⚠ WARNING: Top 3 features have >35% importance! Model may be overfitting.")
            if num_features_used < 15:  # AGGRESSIVE: Was 12, now 15
                print(f"  ⚠ WARNING: Only {num_features_used} features are being used! Expected: ≥15 features.")
            else:
                print(f"  ✓ Feature usage is healthy ({num_features_used} features with non-zero importance)")
        
        # 8. Evaluate model
        train_metrics, train_proba = evaluate_model(model, X_train, y_train, "Training")
        val_metrics, val_proba = evaluate_model(model, X_val, y_val, "Validation")
        test_metrics, test_proba = evaluate_model(model, X_test, y_test, "Test")
        
        # 8.5. Overfitting detection: Compare train/val/test metrics
        print("\n" + "=" * 80)
        print("OVERFITTING DETECTION")
        print("=" * 80)
        
        # Check train/val/test gap
        train_val_acc_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
        train_test_acc_gap = abs(train_metrics['accuracy'] - test_metrics['accuracy'])
        val_test_acc_gap = abs(val_metrics['accuracy'] - test_metrics['accuracy'])
        
        print(f"\nAccuracy gaps:")
        print(f"  Train - Val:   {train_val_acc_gap:.4f} ({train_val_acc_gap*100:.2f}%)")
        print(f"  Train - Test:  {train_test_acc_gap:.4f} ({train_test_acc_gap*100:.2f}%)")
        print(f"  Val - Test:    {val_test_acc_gap:.4f} ({val_test_acc_gap*100:.2f}%)")
        
        if train_val_acc_gap > 0.05:
            print(f"  ⚠ WARNING: Large train-val gap ({train_val_acc_gap*100:.2f}%)! Model may be overfitting.")
        else:
            print(f"  ✓ Train-val gap is acceptable (<5%)")
        
        if train_test_acc_gap > 0.05:
            print(f"  ⚠ WARNING: Large train-test gap ({train_test_acc_gap*100:.2f}%)! Model may be overfitting.")
        else:
            print(f"  ✓ Train-test gap is acceptable (<5%)")
        
        # Check ROC-AUC gaps
        train_val_auc_gap = abs(train_metrics['roc_auc'] - val_metrics['roc_auc'])
        train_test_auc_gap = abs(train_metrics['roc_auc'] - test_metrics['roc_auc'])
        
        print(f"\nROC-AUC gaps:")
        print(f"  Train - Val:   {train_val_auc_gap:.4f}")
        print(f"  Train - Test:  {train_test_auc_gap:.4f}")
        
        if train_val_auc_gap > 0.05:
            print(f"  ⚠ WARNING: Large train-val ROC-AUC gap ({train_val_auc_gap:.4f})! Model may be overfitting.")
        
        # Check for perfect predictions
        if train_metrics.get('unique_probabilities', 0) <= 2:
            print(f"\n  ⚠ WARNING: Training set has only {train_metrics.get('unique_probabilities', 0)} unique probabilities!")
        if val_metrics.get('unique_probabilities', 0) <= 2:
            print(f"  ⚠ WARNING: Validation set has only {val_metrics.get('unique_probabilities', 0)} unique probabilities!")
        if test_metrics.get('unique_probabilities', 0) <= 2:
            print(f"  ⚠ WARNING: Test set has only {test_metrics.get('unique_probabilities', 0)} unique probabilities!")
        
        print("=" * 80)
        
        # 9. Create visualizations
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating visualizations...")
        plot_roc_curve(y_test, test_proba, str(output_path / "roc_curve.png"))
        plot_calibration_curve(y_test, test_proba, str(output_path / "calibration_curve.png"))
        plot_feature_importance(importance_df, top_n=20, save_path=str(output_path / "feature_importance.png"))
        
        # 10. Save model and artifacts
        save_model(
            model,
            feature_engineer,
            best_params,
            test_metrics,
            importance_df,
            args.output
        )
        
        # 11. Save evaluation report
        save_evaluation_report(train_metrics, val_metrics, test_metrics, args.output)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nFinal Test Set Performance:")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.6f}")
        print(f"  F1-Score: {test_metrics['f1_score']:.6f}")
        print(f"  Precision: {test_metrics['precision']:.6f}")
        print(f"  Recall: {test_metrics['recall']:.6f}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

