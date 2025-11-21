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
        # Set random seed for this trial
        trial_seed = random_state + trial.number
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': random_state,
            'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 2),  # Keep at 2
            'min_child_weight': trial.suggest_int('min_child_weight', 50, 100),  # PHASE 0: Much higher: 50-100
            'subsample': trial.suggest_float('subsample', 0.5, 0.7),  # PHASE 0: Lower: 0.5-0.7
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.45),  # PHASE 0: Lower: 0.3-0.45 to force feature diversity
            'reg_alpha': trial.suggest_float('reg_alpha', 100, 200),  # PHASE 0: Much higher: 100-200
            'reg_lambda': trial.suggest_float('reg_lambda', 100, 200),  # PHASE 0: Much higher: 100-200
            'gamma': trial.suggest_float('gamma', 10, 30),  # PHASE 0: Higher: 10-30
            'scale_pos_weight': scale_pos_weight,
        }
        
        # PHASE 0: Add much more noise to training data to break perfect separability
        X_train_noisy = X_train.copy()
        base_noise_scale = 0.25  # PHASE 0: Increased from 0.12 to 0.25
        # Use trial number to vary noise across trials
        np.random.seed(trial_seed)
        for col in X_train_noisy.columns:
            if X_train_noisy[col].dtype in [np.float64, np.int64]:
                col_std = X_train_noisy[col].std()
                if col_std > 0:
                    # Apply 2x noise to mood_score and step_count (symptom features)
                    if 'mood_score' in col.lower() or 'step_count' in col.lower():
                        noise_scale = base_noise_scale * 2.0
                    else:
                        noise_scale = base_noise_scale
                    noise = np.random.normal(0, noise_scale * col_std, size=len(X_train_noisy))
                    X_train_noisy[col] = X_train_noisy[col] + noise
        
        # Create DMatrix for XGBoost with feature names
        dtrain = xgb.DMatrix(X_train_noisy, label=y_train, feature_names=X_train.columns.tolist())
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
        
        # Train model with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # PHASE 0: Check feature importance constraint
        importance_dict = model.get_score(importance_type='gain')
        if importance_dict:
            importance_values = list(importance_dict.values())
            if importance_values:
                total_importance = sum(importance_values)
                if total_importance > 0:
                    sorted_importance = sorted(importance_values, reverse=True)
                    top3_importance = sum(sorted_importance[:3])
                    top3_pct = top3_importance / total_importance
                    
                    # Reject if top 3 features > 50% of total importance
                    if top3_pct > 0.50:
                        return float('inf')  # Reject this trial
                    
                    # Count features with non-zero importance
                    num_features_used = sum(1 for v in importance_values if v > 0)
                    # Reject if < 12 features used
                    if num_features_used < 12:
                        return float('inf')  # Reject this trial
                    
                    # PHASE 0: Check if symptom features dominate (>40% of top 3)
                    # Get feature names from importance dict
                    feature_names_list = X_train.columns.tolist()
                    symptom_keywords = ['mood_score', 'step_count', 'screen_brightness']
                    
                    # Get top 3 feature names and check if they're symptom features
                    if feature_names_list and len(sorted_importance) >= 3:
                        # Get top 3 feature names
                        top3_feature_names = []
                        for feat_name, importance_val in importance_dict.items():
                            if isinstance(feat_name, str):
                                if feat_name.startswith('f'):
                                    # f0, f1 format - get index
                                    try:
                                        idx = int(feat_name[1:])
                                        if idx < len(feature_names_list):
                                            top3_feature_names.append((feature_names_list[idx], importance_val))
                                    except:
                                        pass
                                else:
                                    # Actual feature name
                                    top3_feature_names.append((feat_name, importance_val))
                        
                        # Sort by importance and get top 3
                        top3_feature_names.sort(key=lambda x: x[1], reverse=True)
                        top3_names = [name for name, _ in top3_feature_names[:3]]
                        
                        # Check if symptom features are in top 3
                        symptom_in_top3 = [name for name in top3_names 
                                         if any(keyword in name.lower() for keyword in symptom_keywords)]
                        
                        if symptom_in_top3:
                            # Calculate symptom importance in top 3
                            symptom_importance = sum(imp for name, imp in top3_feature_names[:3] 
                                                   if any(keyword in name.lower() for keyword in symptom_keywords))
                            symptom_top3_pct = symptom_importance / top3_importance if top3_importance > 0 else 0
                            
                            # Reject if symptom features > 40% of top 3
                            if symptom_top3_pct > 0.40:
                                return float('inf')  # Reject this trial
        
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
    
    # PHASE 0: Add much more noise to training data to break perfect separability
    print("Adding noise to training data to prevent overfitting...")
    X_train_noisy = X_train.copy()
    base_noise_scale = 0.25  # PHASE 0: Increased from 0.12 to 0.25
    np.random.seed(42)  # For reproducibility
    for col in X_train_noisy.columns:
        if X_train_noisy[col].dtype in [np.float64, np.int64]:
            col_std = X_train_noisy[col].std()
            if col_std > 0:
                # Apply 2x noise to mood_score and step_count (symptom features)
                if 'mood_score' in col.lower() or 'step_count' in col.lower():
                    noise_scale = base_noise_scale * 2.0
                else:
                    noise_scale = base_noise_scale
                noise = np.random.normal(0, noise_scale * col_std, size=len(X_train_noisy))
                X_train_noisy[col] = X_train_noisy[col] + noise
    
    # Create DMatrix with feature names
    dtrain = xgb.DMatrix(X_train_noisy, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
    
    # Train model
    print("Training XGBoost model...")
    model = xgb.train(
        hyperparameters,
        dtrain,
        num_boost_round=min(n_estimators, 100),  # Much lower cap: 100 instead of 200
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10,  # More aggressive early stopping
        verbose_eval=25
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

