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
        
        # AGGRESSIVE: Remove mood_category - it's a perfect predictor derived from mood_score
        if 'mood_category' in X_processed.columns:
            print("Removing 'mood_category' feature (creates perfect separability - derived from mood_score)")
            X_processed = X_processed.drop(columns=['mood_category'])
            self.removed_features.append('mood_category')
        
        # AGGRESSIVE: Consider removing symptom features entirely if they're causing overfitting
        # We'll keep them but add heavy noise during training instead
        # This allows the model to learn from predictors while symptoms are heavily noised
        
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
        
        # PHASE 0: Remove mood_score_binned - it creates perfect separability
        # Do NOT create mood_score_binned - it's a derived feature that amplifies symptom signal
        
        # PHASE 0: Remove symptom-based interactions - they amplify symptom signal
        # Instead, create predictor-based interactions (see below)
        
        # Create PREDICTOR-based interaction features (not symptom-based)
        # These help the model use logical predictor features
        # PHASE 3: Normalize all interactions to prevent dominance
        if 'stress_intensity' in X_processed.columns and 'sleep' in X_processed.columns:
            # Normalize before interaction
            stress_norm = (X_processed['stress_intensity'] - X_processed['stress_intensity'].mean()) / (X_processed['stress_intensity'].std() + 1e-8)
            sleep_norm = (X_processed['sleep'] - X_processed['sleep'].mean()) / (X_processed['sleep'].std() + 1e-8)
            X_processed['stress_sleep_interaction'] = stress_norm * sleep_norm
            print("Created stress_sleep_interaction feature (normalized: stress_intensity × sleep)")
        
        if 'weather' in X_processed.columns and 'pressure_mean' in X_processed.columns:
            # Normalize pressure deviation for interaction
            pressure_mean = X_processed['pressure_mean'].mean()
            pressure_deviation = abs(X_processed['pressure_mean'] - pressure_mean) / 20.0
            weather_norm = (X_processed['weather'] - X_processed['weather'].mean()) / (X_processed['weather'].std() + 1e-8)
            pressure_dev_norm = (pressure_deviation - pressure_deviation.mean()) / (pressure_deviation.std() + 1e-8)
            X_processed['weather_pressure_interaction'] = weather_norm * pressure_dev_norm
            print("Created weather_pressure_interaction feature (normalized: weather × pressure_deviation)")
        
        if 'stress_intensity' in X_processed.columns:
            # Aggregate trigger features if available
            trigger_cols = ['stress', 'hormonal', 'sleep', 'weather', 'food', 'sensory', 'physical']
            available_triggers = [col for col in trigger_cols if col in X_processed.columns]
            if available_triggers:
                trigger_aggregate = X_processed[available_triggers].sum(axis=1)
                # Normalize aggregate
                trigger_agg_norm = (trigger_aggregate - trigger_aggregate.mean()) / (trigger_aggregate.std() + 1e-8)
                X_processed['trigger_aggregate'] = trigger_agg_norm
                
                # Normalize stress for interaction
                stress_norm = (X_processed['stress_intensity'] - X_processed['stress_intensity'].mean()) / (X_processed['stress_intensity'].std() + 1e-8)
                X_processed['stress_trigger_interaction'] = stress_norm * trigger_agg_norm
                print(f"Created trigger_aggregate and stress_trigger_interaction features (normalized)")
        
        if 'pressure_mean' in X_processed.columns and 'temp_mean' in X_processed.columns:
            # Normalize for interaction
            pressure_norm = (X_processed['pressure_mean'] - X_processed['pressure_mean'].mean()) / (X_processed['pressure_mean'].std() + 1e-8)
            temp_norm = (X_processed['temp_mean'] - X_processed['temp_mean'].mean()) / (X_processed['temp_mean'].std() + 1e-8)
            X_processed['pressure_temp_interaction'] = pressure_norm * temp_norm
            print("Created pressure_temp_interaction feature (normalized: pressure_mean × temp_mean)")
        
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
        
        # PHASE 3: Track feature groups for balancing
        predictor_features = [col for col in self.feature_names if col in [
            'stress_intensity', 'temp_mean', 'wind_mean', 'pressure_mean',
            'sun_irr_mean', 'sun_time_mean', 'precip_total', 'cloud_mean',
            'stress', 'hormonal', 'sleep', 'weather', 'food', 'sensory', 'physical'
        ]]
        temporal_features = [col for col in self.feature_names if col in [
            'consecutive_migraine_days', 'days_since_last_migraine'
        ]]
        symptom_features = [col for col in self.feature_names if any(symptom in col.lower() for symptom in [
            'mood_score', 'step_count', 'screen_brightness'
        ])]
        interaction_features = [col for col in self.feature_names if 'interaction' in col.lower() or 'aggregate' in col.lower()]
        
        self.feature_groups = {
            'predictor': predictor_features,
            'temporal': temporal_features,
            'symptom': symptom_features,
            'interaction': interaction_features,
            'other': [col for col in self.feature_names if col not in 
                predictor_features + temporal_features + symptom_features + interaction_features]
        }
        
        print(f"\nFinal feature count: {len(self.feature_names)}")
        print(f"Removed features: {self.removed_features}")
        print(f"\nFeature Groups:")
        for group_name, group_features in self.feature_groups.items():
            if group_features:
                print(f"  {group_name.capitalize()}: {len(group_features)} features")
        
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
        
        # PHASE 0: Do NOT recreate mood_score_binned or symptom interactions
        # PHASE 3: Recreate predictor-based interactions with normalization (same as fit_transform)
        if 'stress_intensity' in X_processed.columns and 'sleep' in X_processed.columns:
            stress_norm = (X_processed['stress_intensity'] - X_processed['stress_intensity'].mean()) / (X_processed['stress_intensity'].std() + 1e-8)
            sleep_norm = (X_processed['sleep'] - X_processed['sleep'].mean()) / (X_processed['sleep'].std() + 1e-8)
            X_processed['stress_sleep_interaction'] = stress_norm * sleep_norm
        
        if 'weather' in X_processed.columns and 'pressure_mean' in X_processed.columns:
            pressure_mean = X_processed['pressure_mean'].mean()
            pressure_deviation = abs(X_processed['pressure_mean'] - pressure_mean) / 20.0
            weather_norm = (X_processed['weather'] - X_processed['weather'].mean()) / (X_processed['weather'].std() + 1e-8)
            pressure_dev_norm = (pressure_deviation - pressure_deviation.mean()) / (pressure_deviation.std() + 1e-8)
            X_processed['weather_pressure_interaction'] = weather_norm * pressure_dev_norm
        
        if 'stress_intensity' in X_processed.columns:
            trigger_cols = ['stress', 'hormonal', 'sleep', 'weather', 'food', 'sensory', 'physical']
            available_triggers = [col for col in trigger_cols if col in X_processed.columns]
            if available_triggers:
                trigger_aggregate = X_processed[available_triggers].sum(axis=1)
                trigger_agg_norm = (trigger_aggregate - trigger_aggregate.mean()) / (trigger_aggregate.std() + 1e-8)
                X_processed['trigger_aggregate'] = trigger_agg_norm
                
                stress_norm = (X_processed['stress_intensity'] - X_processed['stress_intensity'].mean()) / (X_processed['stress_intensity'].std() + 1e-8)
                X_processed['stress_trigger_interaction'] = stress_norm * trigger_agg_norm
        
        if 'pressure_mean' in X_processed.columns and 'temp_mean' in X_processed.columns:
            pressure_norm = (X_processed['pressure_mean'] - X_processed['pressure_mean'].mean()) / (X_processed['pressure_mean'].std() + 1e-8)
            temp_norm = (X_processed['temp_mean'] - X_processed['temp_mean'].mean()) / (X_processed['temp_mean'].std() + 1e-8)
            X_processed['pressure_temp_interaction'] = pressure_norm * temp_norm
        
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
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for balancing analysis."""
        return self.feature_groups.copy() if hasattr(self, 'feature_groups') else {}

