"""
Model evaluation module.
Comprehensive metrics and visualizations.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report,
    brier_score_loss, accuracy_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(
    model: xgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    set_name: str = "Test"
) -> Dict:
    """
    Evaluate model performance.
    
    Args:
        model: Trained XGBoost model
        X: Features
        y: True labels
        set_name: Name of the dataset (for reporting)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n" + "=" * 80)
    print(f"MODEL EVALUATION - {set_name.upper()} SET")
    print("=" * 80)
    
    # Create DMatrix with feature names
    dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
    
    # Get predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Convert y to int if needed
    if y.dtype == bool:
        y_int = y.astype(int)
    else:
        y_int = y
    
    # Calculate metrics first
    metrics = {
        'roc_auc': roc_auc_score(y_int, y_pred_proba),
        'log_loss': log_loss(y_int, y_pred_proba),
        'brier_score': brier_score_loss(y_int, y_pred_proba),
        'precision': precision_score(y_int, y_pred),
        'recall': recall_score(y_int, y_pred),
        'f1_score': f1_score(y_int, y_pred),
        'accuracy': accuracy_score(y_int, y_pred),
    }
    
    # Validation checks for suspicious predictions (after metrics are calculated)
    unique_proba = len(np.unique(y_pred_proba))
    unique_pred = len(np.unique(y_pred))
    proba_range = y_pred_proba.max() - y_pred_proba.min()
    
    # Enhanced overfitting detection
    print(f"\nOverfitting Detection:")
    print(f"  Unique probability values: {unique_proba}")
    print(f"  Probability range: [{y_pred_proba.min():.6f}, {y_pred_proba.max():.6f}]")
    
    if unique_proba <= 2:
        print(f"⚠ WARNING: Only {unique_proba} unique probability values! Model may be broken.")
        print(f"   This suggests perfect separability or severe overfitting.")
        print(f"   Expected: >100 unique values for a healthy model.")
    elif unique_proba < 100:
        print(f"⚠ WARNING: Low number of unique probability values ({unique_proba}).")
        print(f"   This may indicate overfitting. Expected: >100 unique values.")
    else:
        print(f"✓ Number of unique probability values is healthy ({unique_proba})")
    
    if unique_pred == 1:
        print(f"⚠ WARNING: Model predicts only one class ({y_pred[0]})! Severe overfitting or data issue.")
        print(f"   All predictions: {y_pred[0]}")
        print(f"   Probability range: [{y_pred_proba.min():.6f}, {y_pred_proba.max():.6f}]")
    if (y_pred_proba == 1.0).all() or (y_pred_proba == 0.0).all():
        print(f"⚠ WARNING: All probabilities are the same ({y_pred_proba[0]:.6f})! Model may be broken.")
    if proba_range < 0.1:
        print(f"⚠ WARNING: Probability range is very small ({proba_range:.6f})! Model may be overconfident.")
    if metrics['log_loss'] < 0.01:
        print(f"⚠ WARNING: Log loss is extremely low ({metrics['log_loss']:.6f})! Model may be overfitting.")
    
    # Store overfitting indicators in metrics
    metrics['unique_probabilities'] = unique_proba
    metrics['probability_range'] = proba_range
    
    # Confusion matrix
    cm = confusion_matrix(y_int, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"  ROC-AUC Score:    {metrics['roc_auc']:.6f}")
    print(f"  Log Loss:         {metrics['log_loss']:.6f}")
    print(f"  Brier Score:      {metrics['brier_score']:.6f}")
    print(f"  Precision:        {metrics['precision']:.6f}")
    print(f"  Recall:           {metrics['recall']:.6f}")
    print(f"  F1-Score:         {metrics['f1_score']:.6f}")
    print(f"  Accuracy:         {metrics['accuracy']:.6f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              No      Yes")
    print(f"Actual No   {metrics['tn']:5d}   {metrics['fp']:5d}")
    print(f"       Yes  {metrics['fn']:5d}   {metrics['tp']:5d}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_int, y_pred, target_names=['No Migraine', 'Migraine']))
    
    print("=" * 80)
    
    return metrics, y_pred_proba


def plot_roc_curve(y_true: pd.Series, y_pred_proba: np.ndarray, save_path: str = None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_calibration_curve(y_true: pd.Series, y_pred_proba: np.ndarray, save_path: str = None):
    """
    Plot calibration curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Model', linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label='Perfectly Calibrated', linewidth=1)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, save_path: str = None):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to plot
        save_path: Path to save plot
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

