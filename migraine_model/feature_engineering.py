"""
Feature engineering module.
Handles categorical encoding and feature validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering pipeline."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.label_encoders = {}
        self.feature_names = []
        self.removed_features = []
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoders and transform features.
        
        Args:
            X: Input features DataFrame
        
        Returns:
            Transformed features DataFrame
        """
        X_processed = X.copy()
        
        # Remove constant features
        constant_features = []
        for col in X_processed.columns:
            if X_processed[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"Removing constant features: {constant_features}")
            X_processed = X_processed.drop(columns=constant_features)
            self.removed_features.extend(constant_features)
        
        # Encode categorical features
        categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col == 'gender':
                # Binary encoding for gender
                X_processed[col] = (X_processed[col] == 'female').astype(int)
                print(f"Encoded {col}: male=0, female=1")
            
            elif col == 'mood_category':
                # Ordinal encoding for mood_category
                mood_order = ['Very Low', 'Low', 'Moderate', 'Good', 'Very Good']
                mood_mapping = {val: idx for idx, val in enumerate(mood_order)}
                X_processed[col] = X_processed[col].map(mood_mapping)
                print(f"Encoded {col}: {mood_mapping}")
            
            else:
                # Generic label encoding for other categoricals
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
                print(f"Label encoded {col}")
        
        # Convert boolean columns to int
        bool_cols = X_processed.select_dtypes(include=['bool']).columns.tolist()
        for col in bool_cols:
            X_processed[col] = X_processed[col].astype(int)
        
        # Validate numeric ranges for normalized features
        normalized_features = ['step_count_normalized', 'screen_brightness_normalized']
        for col in normalized_features:
            if col in X_processed.columns:
                min_val = X_processed[col].min()
                max_val = X_processed[col].max()
                if min_val < 0 or max_val > 1:
                    print(f"⚠ Warning: {col} is outside [0,1] range: [{min_val:.4f}, {max_val:.4f}]")
        
        # Ensure all features are numeric
        non_numeric = X_processed.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"⚠ Warning: Non-numeric features remaining: {non_numeric}")
        
        self.feature_names = X_processed.columns.tolist()
        
        print(f"\nFinal feature count: {len(self.feature_names)}")
        print(f"Removed features: {self.removed_features}")
        
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted encoders.
        
        Args:
            X: Input features DataFrame
        
        Returns:
            Transformed features DataFrame
        """
        X_processed = X.copy()
        
        # Remove same constant features
        if self.removed_features:
            X_processed = X_processed.drop(columns=[col for col in self.removed_features if col in X_processed.columns])
        
        # Apply same encodings
        if 'gender' in X_processed.columns:
            X_processed['gender'] = (X_processed['gender'] == 'female').astype(int)
        
        if 'mood_category' in X_processed.columns:
            mood_order = ['Very Low', 'Low', 'Moderate', 'Good', 'Very Good']
            mood_mapping = {val: idx for idx, val in enumerate(mood_order)}
            X_processed['mood_category'] = X_processed['mood_category'].map(mood_mapping)
        
        # Apply label encoders
        for col, le in self.label_encoders.items():
            if col in X_processed.columns:
                X_processed[col] = le.transform(X_processed[col].astype(str))
        
        # Convert boolean to int
        bool_cols = X_processed.select_dtypes(include=['bool']).columns.tolist()
        for col in bool_cols:
            X_processed[col] = X_processed[col].astype(int)
        
        # Ensure same feature order
        X_processed = X_processed[self.feature_names]
        
        return X_processed
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation."""
        return self.feature_names.copy()

