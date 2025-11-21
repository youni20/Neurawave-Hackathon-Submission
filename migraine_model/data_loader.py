"""
Data loading and validation module.
Loads CSV file and validates data quality.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV file with all features.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame with all data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def validate_data(df: pd.DataFrame) -> Dict:
    """
    Validate data quality and return validation report.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "=" * 80)
    print("DATA VALIDATION")
    print("=" * 80)
    
    validation_report = {
        'shape': df.shape,
        'missing_values': {},
        'data_types': {},
        'target_distribution': {},
        'categorical_features': [],
        'numeric_features': [],
        'constant_features': [],
        'class_imbalance_ratio': None
    }
    
    # Check for missing values
    missing = df.isnull().sum()
    validation_report['missing_values'] = missing[missing > 0].to_dict()
    if len(validation_report['missing_values']) == 0:
        print("✓ No missing values found")
    else:
        print(f"⚠ Missing values found in {len(validation_report['missing_values'])} columns")
        print(validation_report['missing_values'])
    
    # Data types
    validation_report['data_types'] = df.dtypes.to_dict()
    print(f"\nData types: {df.dtypes.value_counts().to_dict()}")
    
    # Identify categorical and numeric features
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    validation_report['categorical_features'] = [col for col in categorical_cols if col != 'migraine']
    validation_report['numeric_features'] = numeric_cols
    
    print(f"\nCategorical features: {validation_report['categorical_features']}")
    print(f"Numeric features: {len(validation_report['numeric_features'])} features")
    
    # Check target distribution
    if 'migraine' in df.columns:
        target_counts = df['migraine'].value_counts()
        target_dist = df['migraine'].value_counts(normalize=True)
        
        validation_report['target_distribution'] = {
            'counts': target_counts.to_dict(),
            'proportions': target_dist.to_dict(),
            'total': len(df)
        }
        
        # Calculate class imbalance ratio
        if len(target_counts) == 2:
            negative_count = target_counts.get(False, target_counts.get(0, 0))
            positive_count = target_counts.get(True, target_counts.get(1, 0))
            if positive_count > 0:
                validation_report['class_imbalance_ratio'] = negative_count / positive_count
        
        print(f"\nTarget distribution (migraine):")
        print(f"  False/No: {target_counts.get(False, target_counts.get(0, 0)):,} ({target_dist.get(False, target_dist.get(0, 0))*100:.2f}%)")
        print(f"  True/Yes: {target_counts.get(True, target_counts.get(1, 0)):,} ({target_dist.get(True, target_dist.get(1, 0))*100:.2f}%)")
        
        if validation_report['class_imbalance_ratio']:
            print(f"  Class imbalance ratio: {validation_report['class_imbalance_ratio']:.2f}")
    
    # Check for constant features
    constant_features = []
    for col in df.columns:
        if col != 'migraine':
            if df[col].nunique() <= 1:
                constant_features.append(col)
    
    validation_report['constant_features'] = constant_features
    if constant_features:
        print(f"\n⚠ Constant features found: {constant_features}")
    else:
        print("\n✓ No constant features found")
    
    # Check for perfect predictors (features that perfectly separate classes)
    if 'migraine' in df.columns:
        perfect_predictors = []
        for col in df.columns:
            if col != 'migraine' and df[col].dtype in [np.number, 'object']:
                if df[col].dtype == 'object':
                    # For categorical, check each value
                    for val in df[col].unique():
                        mask = df[col] == val
                        if mask.sum() > 10:  # Only check if enough samples
                            target_rate = df.loc[mask, 'migraine'].mean()
                            if target_rate == 0.0 or target_rate == 1.0:
                                perfect_predictors.append(f"{col}={val}")
                                break
                else:
                    # For numeric, check if correlation is too high
                    corr = abs(df[col].corr(df['migraine'].astype(int)))
                    if corr > 0.95:
                        perfect_predictors.append(f"{col} (corr={corr:.3f})")
        
        if perfect_predictors:
            print(f"\n⚠ WARNING: Perfect or near-perfect predictors found!")
            print(f"   These features may cause severe overfitting:")
            for pred in perfect_predictors[:10]:  # Show first 10
                print(f"     - {pred}")
            if len(perfect_predictors) > 10:
                print(f"     ... and {len(perfect_predictors) - 10} more")
            print(f"   The model will use noise injection and aggressive regularization to mitigate this.")
        else:
            print("\n✓ No perfect predictors found")
    
    # Basic statistics
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print("=" * 80)
    
    return validation_report


def prepare_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.
    
    Args:
        df: DataFrame with features and target
    
    Returns:
        Tuple of (features_df, target_series)
    """
    if 'migraine' not in df.columns:
        raise ValueError("Target column 'migraine' not found in dataframe")
    
    X = df.drop(columns=['migraine'])
    y = df['migraine']
    
    # Check for data leakage
    if 'migraine' in X.columns:
        raise ValueError("ERROR: Target 'migraine' found in features! Data leakage detected!")
    
    # Convert boolean to int if needed
    if y.dtype == bool:
        y = y.astype(int)
    
    # Validate target has both classes
    unique_targets = y.unique()
    if len(unique_targets) < 2:
        raise ValueError(f"ERROR: Target has only {len(unique_targets)} unique value(s): {unique_targets}. Need at least 2 classes!")
    
    return X, y

