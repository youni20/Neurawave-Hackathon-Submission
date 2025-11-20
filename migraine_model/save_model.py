"""
Model persistence module.
Save model, encoders, and metadata.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, Any


def save_model(
    model: xgb.Booster,
    feature_engineer: Any,
    hyperparameters: Dict,
    metrics: Dict,
    feature_importance: pd.DataFrame,
    output_dir: str
):
    """
    Save trained model and all related artifacts.
    
    Args:
        model: Trained XGBoost model
        feature_engineer: Feature engineering pipeline
        hyperparameters: Model hyperparameters
        metrics: Evaluation metrics
        feature_importance: Feature importance DataFrame
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    # Save XGBoost model (JSON format for better compatibility)
    model_path = output_path / "migraine_model.json"
    model.save_model(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
    # Save feature engineer
    feature_engineer_path = output_path / "feature_engineer.pkl"
    with open(feature_engineer_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    print(f"✓ Feature engineer saved to {feature_engineer_path}")
    
    # Save feature names
    feature_names_path = output_path / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(feature_engineer.get_feature_names(), f, indent=2)
    print(f"✓ Feature names saved to {feature_names_path}")
    
    # Save feature importance
    importance_path = output_path / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to {importance_path}")
    
    # Save metadata
    metadata = {
        'hyperparameters': hyperparameters,
        'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                   for k, v in metrics.items() if k != 'confusion_matrix'},
        'confusion_matrix': metrics['confusion_matrix'].tolist() if 'confusion_matrix' in metrics else None,
        'feature_count': len(feature_engineer.get_feature_names()),
        'training_timestamp': datetime.now().isoformat(),
        'model_type': 'XGBoost',
        'objective': 'binary:logistic',
    }
    
    metadata_path = output_path / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Model metadata saved to {metadata_path}")
    
    print("=" * 80)
    print(f"\nAll model artifacts saved to: {output_path.absolute()}")


def save_evaluation_report(
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    output_dir: str
):
    """
    Save comprehensive evaluation report.
    
    Args:
        train_metrics: Training set metrics
        val_metrics: Validation set metrics
        test_metrics: Test set metrics
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    report_path = output_path / "model_evaluation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MIGRAINE PREDICTION MODEL - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for set_name, metrics in [("Training", train_metrics), 
                                  ("Validation", val_metrics), 
                                  ("Test", test_metrics)]:
            f.write(f"\n{set_name} Set Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"ROC-AUC Score:    {metrics.get('roc_auc', 0):.6f}\n")
            f.write(f"Log Loss:         {metrics.get('log_loss', 0):.6f}\n")
            f.write(f"Brier Score:      {metrics.get('brier_score', 0):.6f}\n")
            f.write(f"Precision:        {metrics.get('precision', 0):.6f}\n")
            f.write(f"Recall:           {metrics.get('recall', 0):.6f}\n")
            f.write(f"F1-Score:         {metrics.get('f1_score', 0):.6f}\n")
            f.write(f"Accuracy:         {metrics.get('accuracy', 0):.6f}\n")
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                f.write(f"\nConfusion Matrix:\n")
                f.write(f"                Predicted\n")
                f.write(f"              No      Yes\n")
                f.write(f"Actual No   {metrics['tn']:5d}   {metrics['fp']:5d}\n")
                f.write(f"       Yes  {metrics['fn']:5d}   {metrics['tp']:5d}\n")
            f.write("\n")
    
    print(f"✓ Evaluation report saved to {report_path}")

