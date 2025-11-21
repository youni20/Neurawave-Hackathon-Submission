"""
XGBoost model training with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import warnings
warnings.filterwarnings('ignore')


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for class imbalance.
    
    Args:
        y: Target series
    
    Returns:
        Scale positive weight value
    """
    negative_count = (y == 0).sum() if y.dtype != bool else (y == False).sum()
    positive_count = (y == 1).sum() if y.dtype != bool else (y == True).sum()
    
    if positive_count > 0:
        return negative_count / positive_count
    return 1.0


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    random_state: int = 42
) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
        random_state: Random seed
    
    Returns:
        Dictionary of best hyperparameters
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Running {n_trials} optimization trials...")
    
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.4f}")
    
    def objective(trial):
        """Objective function for Optuna."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': random_state,
            'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),  # Reduced from 5-15 to prevent overfitting
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),  # Increased from 1-10
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),  # Reduced max from 1.0
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),  # Reduced max from 1.0
            'reg_alpha': trial.suggest_float('reg_alpha', 1, 15),  # Increased from 0-10
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 15),  # Increased from 0-10
            'gamma': trial.suggest_float('gamma', 0, 5),  # Add gamma for additional regularization
            'scale_pos_weight': scale_pos_weight,
        }
        
        # Create DMatrix for XGBoost with feature names
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
        
        # Train model with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,  # Reduced from 1500
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,  # Increased from 50 for better early stopping
            verbose_eval=False
        )
        
        # Get best score
        best_score = model.best_score
        return best_score
    
    # Create study and optimize
    study = optuna.create_study(
        direction='minimize',  # Minimize logloss
        study_name='xgboost_migraine',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': random_state,
        'n_jobs': -1,
        'scale_pos_weight': scale_pos_weight,
    })
    
    print(f"\nBest trial:")
    print(f"  Value (logloss): {study.best_value:.6f}")
    print(f"  Params:")
    for key, value in best_params.items():
        if key not in ['objective', 'eval_metric', 'tree_method', 'random_state', 'n_jobs']:
            print(f"    {key}: {value}")
    
    print("=" * 80)
    
    return best_params


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparameters: Dict,
    n_estimators: int = 1000
) -> xgb.Booster:
    """
    Train final XGBoost model with best hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        hyperparameters: Best hyperparameters from optimization
        n_estimators: Maximum number of boosting rounds
    
    Returns:
        Trained XGBoost model
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    # Create DMatrix with feature names
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
    
    # Train model
    print("Training XGBoost model...")
    model = xgb.train(
        hyperparameters,
        dtrain,
        num_boost_round=min(n_estimators, 500),  # Cap at 500 to prevent overfitting
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,  # Increased from 50
        verbose_eval=50
    )
    
    print(f"\nTraining completed!")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.6f}")
    print("=" * 80)
    
    return model


def get_feature_importance(model: xgb.Booster, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    # Get importance scores - use feature names directly if available
    importance_dict = model.get_score(importance_type='gain')
    
    # Check if feature names are in the importance dict (when feature_names were provided to DMatrix)
    if importance_dict and any(name in importance_dict for name in feature_names):
        # Feature names were provided, use them directly
        importance_values = [importance_dict.get(name, 0) for name in feature_names]
    else:
        # Fallback: use f0, f1, f2... format
        importance_values = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df

