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
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 80)
    print("MIGRAINE PREDICTION MODEL TRAINING")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Hyperparameter trials: {args.trials}")
    print(f"Random state: {args.random_state}")
    print("=" * 80)
    
    try:
        # Set random seeds
        np.random.seed(args.random_state)
        
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
            n_estimators=1500
        )
        
        # 7. Get feature importance
        feature_names = feature_engineer.get_feature_names()
        importance_df = get_feature_importance(model, feature_names)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # 8. Evaluate model
        train_metrics, train_proba = evaluate_model(model, X_train, y_train, "Training")
        val_metrics, val_proba = evaluate_model(model, X_val, y_val, "Validation")
        test_metrics, test_proba = evaluate_model(model, X_test, y_test, "Test")
        
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

