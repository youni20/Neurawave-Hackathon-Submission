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
        
        # Remove mood_category - it's a perfect predictor derived from mood_score
        if 'mood_category' in X_processed.columns:
            print("Removing 'mood_category' feature (creates perfect separability - derived from mood_score)")
            X_processed = X_processed.drop(columns=['mood_category'])
            self.removed_features.append('mood_category')
        
        # Encode categorical features
        categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col == 'gender':
                # Binary encoding for gender
                X_processed[col] = (X_processed[col] == 'female').astype(int)
                print(f"Encoded {col}: male=0, female=1")
            
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
        
        # Bin mood_score to reduce separability (wider bins)
        if 'mood_score' in X_processed.columns:
            # Create wider bins: 0-4, 4-7, 7-10
            X_processed['mood_score_binned'] = pd.cut(
                X_processed['mood_score'],
                bins=[-np.inf, 4, 7, np.inf],
                labels=[0, 1, 2]
            ).astype(float)
            print("Created mood_score_binned feature (wider bins to reduce separability)")
        
        # Create feature interactions to force multi-feature usage
        if 'mood_score' in X_processed.columns and 'step_count_normalized' in X_processed.columns:
            # Interaction feature: mood * step_count
            X_processed['mood_step_interaction'] = (
                X_processed['mood_score'] * X_processed['step_count_normalized']
            )
            print("Created mood_step_interaction feature (mood_score * step_count_normalized)")
            
            # Ratio feature: step_count / (mood_score + 1)
            X_processed['step_mood_ratio'] = (
                X_processed['step_count_normalized'] / (X_processed['mood_score'] + 1)
            )
            print("Created step_mood_ratio feature (step_count_normalized / (mood_score + 1))")
        
        # Create interaction with screen brightness if available
        if 'mood_score' in X_processed.columns and 'screen_brightness_normalized' in X_processed.columns:
            X_processed['mood_brightness_interaction'] = (
                X_processed['mood_score'] * X_processed['screen_brightness_normalized']
            )
            print("Created mood_brightness_interaction feature")
        
        if 'step_count_normalized' in X_processed.columns and 'screen_brightness_normalized' in X_processed.columns:
            X_processed['step_brightness_interaction'] = (
                X_processed['step_count_normalized'] * X_processed['screen_brightness_normalized']
            )
            print("Created step_brightness_interaction feature")
        
        # Ensure all features are numeric
        non_numeric = X_processed.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"⚠ Warning: Non-numeric features remaining: {non_numeric}")
            # Convert remaining non-numeric to numeric if possible
            for col in non_numeric:
                try:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                except:
                    pass
        
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
        
        # Remove mood_category if present
        if 'mood_category' in X_processed.columns:
            X_processed = X_processed.drop(columns=['mood_category'])
        
        # Apply same encodings
        if 'gender' in X_processed.columns:
            X_processed['gender'] = (X_processed['gender'] == 'female').astype(int)
        
        # Apply label encoders
        for col, le in self.label_encoders.items():
            if col in X_processed.columns:
                X_processed[col] = le.transform(X_processed[col].astype(str))
        
        # Convert boolean to int
        bool_cols = X_processed.select_dtypes(include=['bool']).columns.tolist()
        for col in bool_cols:
            X_processed[col] = X_processed[col].astype(int)
        
        # Recreate feature interactions (same as fit_transform)
        if 'mood_score' in X_processed.columns:
            X_processed['mood_score_binned'] = pd.cut(
                X_processed['mood_score'],
                bins=[-np.inf, 4, 7, np.inf],
                labels=[0, 1, 2]
            ).astype(float)
        
        if 'mood_score' in X_processed.columns and 'step_count_normalized' in X_processed.columns:
            X_processed['mood_step_interaction'] = (
                X_processed['mood_score'] * X_processed['step_count_normalized']
            )
            X_processed['step_mood_ratio'] = (
                X_processed['step_count_normalized'] / (X_processed['mood_score'] + 1)
            )
        
        if 'mood_score' in X_processed.columns and 'screen_brightness_normalized' in X_processed.columns:
            X_processed['mood_brightness_interaction'] = (
                X_processed['mood_score'] * X_processed['screen_brightness_normalized']
            )
        
        if 'step_count_normalized' in X_processed.columns and 'screen_brightness_normalized' in X_processed.columns:
            X_processed['step_brightness_interaction'] = (
                X_processed['step_count_normalized'] * X_processed['screen_brightness_normalized']
            )
        
        # Ensure same feature order - only select features that exist
        available_features = [f for f in self.feature_names if f in X_processed.columns]
        missing_features = [f for f in self.feature_names if f not in X_processed.columns]
        if missing_features:
            print(f"⚠ Warning: Some expected features are missing in transform: {missing_features}")
        X_processed = X_processed[available_features]
        
        return X_processed
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation."""
        return self.feature_names.copy()

