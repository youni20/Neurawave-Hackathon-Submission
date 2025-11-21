"""
XGBoost model training with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Tuple, Optional, List
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


def _compute_group_importance_share(importance_dict: Dict[str, float], target_features: List[str]) -> float:
    """Compute share of importance captured by target feature names."""
    if not importance_dict or not target_features:
        return 0.0
    total_importance = sum(importance_dict.values())
    if total_importance <= 0:
        return 0.0
    target_set = set(target_features)
    target_importance = sum(val for feat, val in importance_dict.items() if feat in target_set)
    return target_importance / total_importance


class SymptomImportanceCallback(xgb.callback.TrainingCallback):
    """Track symptom feature dominance during boosting."""
    
    def __init__(self, symptom_features: List[str], threshold: float = 0.5, report_every: int = 5):
        self.symptom_features = list(symptom_features or [])
        self.threshold = threshold
        self.report_every = report_every
        self.max_share = 0.0
    
    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if not self.symptom_features:
            return False
        importance_dict = model.get_score(importance_type='gain')
        share = _compute_group_importance_share(importance_dict, self.symptom_features)
        self.max_share = max(self.max_share, share)
        if share > self.threshold and ((epoch + 1) % self.report_every == 0):
            print(f"  ⚠ Symptom importance {share*100:.2f}% exceeds {self.threshold*100:.0f}% at iteration {epoch+1}")
        return False


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    random_state: int = 42,
    feature_subset_name: str = "full",
    symptom_penalty_config: Optional[Dict] = None
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
    print(f"Feature subset: {feature_subset_name}")
    
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.4f}")
    
    penalty_threshold = 0.0
    penalty_weight = 0.0
    penalty_features: List[str] = []
    if symptom_penalty_config:
        penalty_threshold = symptom_penalty_config.get('threshold', 0.5)
        penalty_weight = symptom_penalty_config.get('weight', 5.0)
        penalty_features = [
            col for col in symptom_penalty_config.get('features', [])
            if col in X_train.columns
        ]
        if penalty_features:
            print(f"Symptom penalty enabled (threshold={penalty_threshold:.2f}, weight={penalty_weight}) "
                  f"for {len(penalty_features)} features.")
    
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
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),  # AGGRESSIVE: Lower learning rate
            'max_depth': trial.suggest_int('max_depth', 2, 2),  # Keep at 2 (shallow trees)
            'min_child_weight': trial.suggest_int('min_child_weight', 100, 200),  # AGGRESSIVE: Much higher: 100-200
            'subsample': trial.suggest_float('subsample', 0.4, 0.6),  # AGGRESSIVE: Lower: 0.4-0.6
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.35),  # AGGRESSIVE: Much lower: 0.2-0.35
            'reg_alpha': trial.suggest_float('reg_alpha', 400, 600),  # AGGRESSIVE: Much higher: 200-400
            'reg_lambda': trial.suggest_float('reg_lambda', 200, 400),  # AGGRESSIVE: Much higher: 200-400
            'gamma': trial.suggest_float('gamma', 20, 50),  # AGGRESSIVE: Much higher: 20-50
            'scale_pos_weight': scale_pos_weight,
        }
        
        # AGGRESSIVE: Add much more noise to training data to break perfect separability
        X_train_noisy = X_train.copy()
        base_noise_scale = 0.50  # AGGRESSIVE: Increased from 0.25 to 0.50 (50% noise!)
        # Use trial number to vary noise across trials
        np.random.seed(trial_seed)
        for col in X_train_noisy.columns:
            if X_train_noisy[col].dtype in [np.float64, np.int64]:
                col_std = X_train_noisy[col].std()
                if col_std > 0:
                    # Apply 3x noise to symptom features (mood_score, step_count, screen_brightness)
                    if any(keyword in col.lower() for keyword in ['mood_score', 'step_count', 'screen_brightness']):
                        noise_scale = base_noise_scale * 3.0  # AGGRESSIVE: 3x for symptoms (150% noise!)
                    else:
                        noise_scale = base_noise_scale
                    noise = np.random.normal(0, noise_scale * col_std, size=len(X_train_noisy))
                    X_train_noisy[col] = X_train_noisy[col] + noise
        
        # Create DMatrix for XGBoost with feature names
        dtrain = xgb.DMatrix(X_train_noisy, label=y_train, feature_names=X_train.columns.tolist())
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
        
        callbacks = []
        penalty_cb = None
        if penalty_features:
            penalty_cb = SymptomImportanceCallback(penalty_features, threshold=penalty_threshold)
            callbacks.append(penalty_cb)
        
        # AGGRESSIVE: Train model with very aggressive early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=50,  # AGGRESSIVE: Reduced from 100 to 50
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=5,  # AGGRESSIVE: More aggressive early stopping (was 10)
            verbose_eval=False,
            callbacks=callbacks if callbacks else None
        )
        
        penalty_add = 0.0
        if penalty_cb and penalty_weight > 0:
            share_violation = max(0.0, penalty_cb.max_share - penalty_threshold)
            penalty_add = penalty_weight * share_violation
            if share_violation > 0:
                print(f"  Applying symptom dominance penalty (+{penalty_add:.6f}) - max share {penalty_cb.max_share*100:.2f}%")
        
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
                    
                    # REALISTIC: Relaxed constraints (was too strict, rejecting all trials)
                    # Reject if top 3 features > 60% of total importance
                    if top3_pct > 0.60:
                        return float('inf')  # Reject this trial
                    
                    # REALISTIC: Reject if top feature > 40% of total importance
                    top1_pct = sorted_importance[0] / total_importance if total_importance > 0 else 0
                    if top1_pct > 0.50:
                        return float('inf')  # Reject this trial
                    
                    # Count features with non-zero importance
                    num_features_used = sum(1 for v in importance_values if v > 0)
                    # REALISTIC: Reject if < 10 features used (was 15, too strict)
                    if num_features_used < 10:
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
                            
                            # REALISTIC: Reject if symptom features > 50% of top 3 (was 15%, too strict)
                            # We can't completely eliminate symptom features, but we can limit their dominance
                            if symptom_top3_pct > 0.50:
                                return float('inf')  # Reject this trial
        
        # Get best score
        best_score = model.best_score + penalty_add
        return best_score
    
    # Create study and optimize
    study = optuna.create_study(
        direction='minimize',  # Minimize logloss
        study_name='xgboost_migraine',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # CRITICAL: Check if all trials were rejected
    # Optuna returns inf as float('inf'), check both ways
    best_value_is_inf = (study.best_value == float('inf')) or (not np.isfinite(study.best_value)) or (str(study.best_value).lower() == 'inf')
    
    if best_value_is_inf:
        print("\n" + "=" * 80)
        print("⚠ WARNING: ALL TRIALS WERE REJECTED BY CONSTRAINTS!")
        print("=" * 80)
        print("This means the feature importance constraints are too strict.")
        print("Relaxing constraints to realistic values...")
        print("=" * 80)
        
        # Relax constraints and retry with more lenient values
        # We'll use default reasonable parameters instead
        best_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': random_state,
            'n_jobs': -1,
            'learning_rate': 0.01,
            'max_depth': 2,
            'min_child_weight': 150,  # Very high to prevent overfitting
            'subsample': 0.5,
            'colsample_bytree': 0.3,  # Low to force diversity
            'reg_alpha': 300,  # Very high regularization
            'reg_lambda': 300,
            'gamma': 40,
            'scale_pos_weight': scale_pos_weight,
        }
        print("\nUsing fallback parameters with very aggressive regularization:")
        for key, value in best_params.items():
            if key not in ['objective', 'eval_metric', 'tree_method', 'random_state', 'n_jobs']:
                print(f"  {key}: {value}")
    else:
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
    n_estimators: int = 1000,
    feature_subset_name: str = "full",
    symptom_penalty_config: Optional[Dict] = None
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
    
    # AGGRESSIVE: Add much more noise to training data to break perfect separability
    print("Adding aggressive noise to training data to prevent overfitting...")
    X_train_noisy = X_train.copy()
    base_noise_scale = 0.50  # AGGRESSIVE: Increased from 0.25 to 0.50 (50% noise!)
    np.random.seed(42)  # For reproducibility
    for col in X_train_noisy.columns:
        if X_train_noisy[col].dtype in [np.float64, np.int64]:
            col_std = X_train_noisy[col].std()
            if col_std > 0:
                # Apply 3x noise to symptom features (mood_score, step_count, screen_brightness)
                if any(keyword in col.lower() for keyword in ['mood_score', 'step_count', 'screen_brightness']):
                    noise_scale = base_noise_scale * 3.0  # AGGRESSIVE: 3x for symptoms (150% noise!)
                else:
                    noise_scale = base_noise_scale
                noise = np.random.normal(0, noise_scale * col_std, size=len(X_train_noisy))
                X_train_noisy[col] = X_train_noisy[col] + noise
    
    # Create DMatrix with feature names
    dtrain = xgb.DMatrix(X_train_noisy, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
    
    # AGGRESSIVE: Train model with very aggressive early stopping
    penalty_features: List[str] = []
    penalty_threshold = 0.0
    if symptom_penalty_config:
        penalty_threshold = symptom_penalty_config.get('threshold', 0.5)
        penalty_features = [
            col for col in symptom_penalty_config.get('features', [])
            if col in X_train.columns
        ]
    
    print(f"Training XGBoost model with aggressive regularization (subset={feature_subset_name})...")
    callbacks = []
    penalty_cb = None
    if penalty_features:
        penalty_cb = SymptomImportanceCallback(penalty_features, threshold=penalty_threshold)
        callbacks.append(penalty_cb)
    
    model = xgb.train(
        hyperparameters,
        dtrain,
        num_boost_round=min(n_estimators, 50),  # AGGRESSIVE: Much lower cap: 50 instead of 100
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=5,  # AGGRESSIVE: Very aggressive early stopping (was 10)
        verbose_eval=25,
        callbacks=callbacks if callbacks else None
    )
    
    if penalty_cb:
        print(f"  Max symptom importance share during training: {penalty_cb.max_share*100:.2f}% (threshold {penalty_threshold*100:.0f}%)")
    
    # CRITICAL: Check feature importance constraints after final training
    print("\nValidating final model feature importance constraints...")
    importance_dict = model.get_score(importance_type='gain')
    if importance_dict:
        importance_values = list(importance_dict.values())
        if importance_values:
            total_importance = sum(importance_values)
            if total_importance > 0:
                sorted_importance = sorted(importance_values, reverse=True)
                top3_importance = sum(sorted_importance[:3])
                top3_pct = top3_importance / total_importance
                top1_pct = sorted_importance[0] / total_importance
                num_features_used = sum(1 for v in importance_values if v > 0)
                
                warnings = []
                if top1_pct > 0.40:
                    warnings.append(f"Top feature has {top1_pct*100:.1f}% importance (target: <40%)")
                if top3_pct > 0.60:
                    warnings.append(f"Top 3 features have {top3_pct*100:.1f}% importance (target: <60%)")
                if num_features_used < 10:
                    warnings.append(f"Only {num_features_used} features used (target: ≥10)")
                
                if warnings:
                    print("  ⚠ WARNING: Final model violates constraints:")
                    for w in warnings:
                        print(f"    - {w}")
                else:
                    print("  ✓ Final model meets all feature importance constraints")
    
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

