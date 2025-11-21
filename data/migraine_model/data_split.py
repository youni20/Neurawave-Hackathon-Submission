"""
Data splitting module.
Stratified train/validation/test split.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform stratified train/validation/test split.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion for test set (default: 0.15)
        val_size: Proportion for validation set (default: 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "=" * 80)
    print("DATA SPLITTING")
    print("=" * 80)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust val_size to account for test set already removed
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Check class distribution in each split
    print(f"\nClass distribution:")
    print(f"  Train -   Migraine: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Val   -   Migraine: {y_val.sum():,} ({y_val.mean()*100:.2f}%)")
    print(f"  Test  -   Migraine: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    
    print("=" * 80)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

