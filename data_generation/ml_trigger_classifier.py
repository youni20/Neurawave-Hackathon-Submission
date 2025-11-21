"""
ML-based trigger classifier for questionnaire data.
Uses machine learning models to classify migraine triggers from symptom data.
Includes comprehensive statistics about the dataset columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.multioutput import MultiOutputClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. ML classification will use rule-based fallback.")


class MLTriggerClassifier:
    """Machine learning-based trigger classifier with statistics."""
    
    TRIGGER_CATEGORIES = [
        'stress',
        'hormonal',
        'sleep',
        'weather',
        'food',
        'sensory',
        'physical'
    ]
    
    def __init__(self, questionnaire_file: Optional[str] = None):
        """
        Initialize ML trigger classifier.
        
        Args:
            questionnaire_file: Path to questionnaire CSV file
        """
        self.questionnaire_data = None
        self.statistics = {}
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        
        if questionnaire_file:
            self.load_questionnaire(questionnaire_file)
    
    def load_questionnaire(self, file_path: str):
        """
        Load questionnaire data from CSV file.
        
        Args:
            file_path: Path to CSV file
        """
        try:
            self.questionnaire_data = pd.read_csv(file_path)
            print(f"Loaded questionnaire data: {len(self.questionnaire_data)} records")
            self._compute_statistics()
        except Exception as e:
            print(f"Error loading questionnaire file {file_path}: {e}")
            self.questionnaire_data = None
    
    def _compute_statistics(self):
        """Compute comprehensive statistics for all columns."""
        if self.questionnaire_data is None or len(self.questionnaire_data) == 0:
            return
        
        self.statistics = {
            'dataset_info': {
                'total_records': len(self.questionnaire_data),
                'total_columns': len(self.questionnaire_data.columns),
                'missing_values': self.questionnaire_data.isnull().sum().to_dict()
            },
            'numeric_columns': {},
            'categorical_columns': {},
            'correlations': {}
        }
        
        # Statistics for numeric columns
        numeric_cols = self.questionnaire_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.statistics['numeric_columns'][col] = {
                'mean': float(self.questionnaire_data[col].mean()),
                'median': float(self.questionnaire_data[col].median()),
                'std': float(self.questionnaire_data[col].std()),
                'min': float(self.questionnaire_data[col].min()),
                'max': float(self.questionnaire_data[col].max()),
                'q25': float(self.questionnaire_data[col].quantile(0.25)),
                'q75': float(self.questionnaire_data[col].quantile(0.75)),
                'missing_count': int(self.questionnaire_data[col].isnull().sum()),
                'missing_percentage': float(self.questionnaire_data[col].isnull().sum() / len(self.questionnaire_data) * 100),
                'unique_values': int(self.questionnaire_data[col].nunique())
            }
        
        # Statistics for categorical columns
        categorical_cols = self.questionnaire_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = self.questionnaire_data[col].value_counts()
            self.statistics['categorical_columns'][col] = {
                'unique_count': int(self.questionnaire_data[col].nunique()),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'most_frequent_percentage': float(value_counts.iloc[0] / len(self.questionnaire_data) * 100) if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.to_dict(),
                'missing_count': int(self.questionnaire_data[col].isnull().sum()),
                'missing_percentage': float(self.questionnaire_data[col].isnull().sum() / len(self.questionnaire_data) * 100)
            }
        
        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = self.questionnaire_data[numeric_cols].corr()
            self.statistics['correlations'] = corr_matrix.to_dict()
    
    def print_statistics(self):
        """Print comprehensive statistics about the dataset."""
        if not self.statistics:
            print("No statistics available. Load questionnaire data first.")
            return
        
        print("=" * 80)
        print("QUESTIONNAIRE DATA STATISTICS")
        print("=" * 80)
        
        # Dataset info
        info = self.statistics['dataset_info']
        print(f"\nDataset Information:")
        print(f"  Total records: {info['total_records']:,}")
        print(f"  Total columns: {info['total_columns']}")
        print(f"  Missing values: {sum(info['missing_values'].values())}")
        
        # Numeric columns statistics
        if self.statistics['numeric_columns']:
            print(f"\n{'=' * 80}")
            print("NUMERIC COLUMNS STATISTICS")
            print(f"{'=' * 80}")
            for col, stats in self.statistics['numeric_columns'].items():
                print(f"\n{col}:")
                print(f"  Mean:        {stats['mean']:.4f}")
                print(f"  Median:      {stats['median']:.4f}")
                print(f"  Std Dev:     {stats['std']:.4f}")
                print(f"  Min:         {stats['min']:.4f}")
                print(f"  Max:         {stats['max']:.4f}")
                print(f"  25th %ile:   {stats['q25']:.4f}")
                print(f"  75th %ile:   {stats['q75']:.4f}")
                print(f"  Unique:      {stats['unique_values']}")
                print(f"  Missing:     {stats['missing_count']} ({stats['missing_percentage']:.2f}%)")
        
        # Categorical columns statistics
        if self.statistics['categorical_columns']:
            print(f"\n{'=' * 80}")
            print("CATEGORICAL COLUMNS STATISTICS")
            print(f"{'=' * 80}")
            for col, stats in self.statistics['categorical_columns'].items():
                print(f"\n{col}:")
                print(f"  Unique values: {stats['unique_count']}")
                print(f"  Most frequent: {stats['most_frequent']} ({stats['most_frequent_count']} times, {stats['most_frequent_percentage']:.2f}%)")
                print(f"  Missing:       {stats['missing_count']} ({stats['missing_percentage']:.2f}%)")
                if len(stats['value_distribution']) <= 10:
                    print(f"  Distribution:")
                    for value, count in list(stats['value_distribution'].items())[:10]:
                        pct = (count / self.statistics['dataset_info']['total_records']) * 100
                        print(f"    {value}: {count} ({pct:.2f}%)")
        
        # Correlation summary
        if self.statistics['correlations']:
            print(f"\n{'=' * 80}")
            print("CORRELATION SUMMARY (Top correlations)")
            print(f"{'=' * 80}")
            # Extract top correlations
            corr_pairs = []
            for col1, corr_dict in self.statistics['correlations'].items():
                for col2, corr_value in corr_dict.items():
                    if col1 != col2 and not np.isnan(corr_value):
                        corr_pairs.append((col1, col2, abs(corr_value)))
            
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            for col1, col2, corr in corr_pairs[:10]:
                print(f"  {col1} <-> {col2}: {corr:.4f}")
        
        print("\n" + "=" * 80)
    
    def _create_trigger_labels(self) -> pd.DataFrame:
        """
        Create trigger labels from questionnaire data using rule-based approach.
        This serves as ground truth for training ML models.
        
        Returns:
            DataFrame with trigger labels (0-1 for each category)
        """
        if self.questionnaire_data is None:
            return pd.DataFrame()
        
        trigger_data = []
        
        for idx, row in self.questionnaire_data.iterrows():
            triggers = {}
            
            # Stress trigger: Based on intensity, frequency, duration
            intensity = row.get('Intensity', 0)
            frequency = row.get('Frequency', 0)
            duration = row.get('Duration', 0)
            stress_score = min((intensity + frequency + duration) / 15.0, 1.0)
            triggers['stress'] = stress_score
            
            # Hormonal trigger: Based on age and intensity
            age = row.get('Age', 30)
            if 20 <= age <= 50:
                triggers['hormonal'] = min(0.3 + (intensity / 10.0) * 0.4, 1.0)
            else:
                triggers['hormonal'] = min(0.1 + (intensity / 10.0) * 0.2, 1.0)
            
            # Sleep trigger: Based on duration
            triggers['sleep'] = min(0.2 + (duration / 5.0) * 0.3, 1.0)
            
            # Weather trigger: Random baseline (would need weather data)
            triggers['weather'] = np.random.beta(1.5, 2.5)
            
            # Food trigger: Based on nausea and vomiting
            nausea = row.get('Nausea', 0)
            vomit = row.get('Vomit', 0)
            triggers['food'] = min((nausea + vomit) / 2.0, 1.0)
            
            # Sensory trigger: Based on photophobia, phonophobia, visual, sensory
            photophobia = row.get('Photophobia', 0)
            phonophobia = row.get('Phonophobia', 0)
            visual = row.get('Visual', 0)
            sensory = row.get('Sensory', 0)
            triggers['sensory'] = min((photophobia + phonophobia + visual + sensory) / 8.0, 1.0)
            
            # Physical trigger: Based on vertigo, ataxia
            vertigo = row.get('Vertigo', 0)
            ataxia = row.get('Ataxia', 0)
            triggers['physical'] = min((vertigo + ataxia) / 2.0, 1.0)
            
            trigger_data.append(triggers)
        
        return pd.DataFrame(trigger_data)
    
    def train_models(self, test_size: float = 0.2, random_state: int = 42):
        """
        Train ML models to classify triggers.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed
        """
        if not HAS_SKLEARN:
            print("scikit-learn not available. Cannot train ML models.")
            return
        
        if self.questionnaire_data is None or len(self.questionnaire_data) == 0:
            print("No questionnaire data loaded. Cannot train models.")
            return
        
        print("Training ML models for trigger classification...")
        
        # Prepare features (exclude Type column if it exists, as it's the outcome)
        feature_cols = [col for col in self.questionnaire_data.columns 
                       if col not in ['Type'] and self.questionnaire_data[col].dtype in [np.int64, np.float64]]
        self.feature_columns = feature_cols
        
        # Create trigger labels
        trigger_labels = self._create_trigger_labels()
        
        # Prepare feature matrix
        X = self.questionnaire_data[feature_cols].fillna(0).values
        y = trigger_labels.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Train models for each trigger category
        model_performances = {}
        
        for idx, trigger_cat in enumerate(self.TRIGGER_CATEGORIES):
            print(f"\nTraining model for {trigger_cat}...")
            
            y_train_cat = y_train[:, idx]
            y_test_cat = y_test[:, idx]
            
            # Convert to binary classification (threshold at 0.5)
            y_train_binary = (y_train_cat >= 0.5).astype(int)
            y_test_binary = (y_test_cat >= 0.5).astype(int)
            
            # Train Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train_binary)
            
            # Evaluate
            y_pred = rf_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_binary, y_pred)
            
            self.models[trigger_cat] = rf_model
            model_performances[trigger_cat] = {
                'accuracy': accuracy,
                'train_samples': len(y_train_binary),
                'test_samples': len(y_test_binary),
                'positive_rate_train': float(y_train_binary.mean()),
                'positive_rate_test': float(y_test_binary.mean())
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Train samples: {len(y_train_binary)}, Test samples: {len(y_test_binary)}")
        
        # Store performance metrics
        self.statistics['model_performance'] = model_performances
        
        print("\n" + "=" * 80)
        print("MODEL TRAINING SUMMARY")
        print("=" * 80)
        for trigger_cat, perf in model_performances.items():
            print(f"{trigger_cat:12s}: Accuracy={perf['accuracy']:.4f}, "
                  f"Train={perf['train_samples']}, Test={perf['test_samples']}")
        print("=" * 80)
    
    def classify_triggers(self, row: pd.Series) -> Dict[str, float]:
        """
        Classify triggers for a single data row using trained ML models.
        
        Args:
            row: Row from questionnaire data
        
        Returns:
            Dictionary of trigger severity scores (0-1) for each category
        """
        if not self.models:
            # Fallback to rule-based if models not trained
            return self._classify_triggers_rule_based(row)
        
        # Prepare features
        feature_values = []
        for col in self.feature_columns:
            value = row.get(col, 0)
            if pd.isna(value):
                value = 0
            feature_values.append(float(value))
        
        feature_array = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        if 'main' in self.scalers:
            feature_array = self.scalers['main'].transform(feature_array)
        
        # Predict for each trigger category
        triggers = {}
        for trigger_cat in self.TRIGGER_CATEGORIES:
            if trigger_cat in self.models:
                # Get probability of positive class
                prob = self.models[trigger_cat].predict_proba(feature_array)[0]
                # Use probability as severity score
                triggers[trigger_cat] = float(prob[1]) if len(prob) > 1 else float(prob[0])
            else:
                triggers[trigger_cat] = 0.0
        
        return triggers
    
    def _classify_triggers_rule_based(self, row: pd.Series) -> Dict[str, float]:
        """Fallback rule-based classification."""
        triggers = {}
        
        intensity = row.get('Intensity', 0)
        frequency = row.get('Frequency', 0)
        duration = row.get('Duration', 0)
        
        triggers['stress'] = min((intensity + frequency + duration) / 15.0, 1.0)
        
        age = row.get('Age', 30)
        if 20 <= age <= 50:
            triggers['hormonal'] = min(0.3 + (intensity / 10.0) * 0.4, 1.0)
        else:
            triggers['hormonal'] = min(0.1 + (intensity / 10.0) * 0.2, 1.0)
        
        triggers['sleep'] = min(0.2 + (duration / 5.0) * 0.3, 1.0)
        triggers['weather'] = np.random.beta(1.5, 2.5)
        
        nausea = row.get('Nausea', 0)
        vomit = row.get('Vomit', 0)
        triggers['food'] = min((nausea + vomit) / 2.0, 1.0)
        
        photophobia = row.get('Photophobia', 0)
        phonophobia = row.get('Phonophobia', 0)
        visual = row.get('Visual', 0)
        sensory = row.get('Sensory', 0)
        triggers['sensory'] = min((photophobia + phonophobia + visual + sensory) / 8.0, 1.0)
        
        vertigo = row.get('Vertigo', 0)
        ataxia = row.get('Ataxia', 0)
        triggers['physical'] = min((vertigo + ataxia) / 2.0, 1.0)
        
        return triggers
    
    def process_dataset(self) -> pd.DataFrame:
        """
        Process entire dataset and classify triggers using ML models.
        
        Returns:
            DataFrame with original data plus trigger classifications
        """
        if self.questionnaire_data is None:
            print("No questionnaire data loaded.")
            return pd.DataFrame()
        
        print("Processing dataset with ML models...")
        
        trigger_data = []
        for idx, row in self.questionnaire_data.iterrows():
            triggers = self.classify_triggers(row)
            trigger_data.append(triggers)
        
        trigger_df = pd.DataFrame(trigger_data)
        
        # Combine with original data
        result_df = pd.concat([self.questionnaire_data, trigger_df], axis=1)
        
        print(f"Processed {len(result_df)} records")
        return result_df
    
    def get_feature_importance(self, trigger_category: str) -> Dict[str, float]:
        """
        Get feature importance for a specific trigger category.
        
        Args:
            trigger_category: Name of trigger category
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if trigger_category not in self.models:
            return {}
        
        model = self.models[trigger_category]
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        importance_dict = {}
        for idx, feature_name in enumerate(self.feature_columns):
            importance_dict[feature_name] = float(model.feature_importances_[idx])
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def print_feature_importance(self, top_n: int = 10):
        """Print feature importance for all trigger categories."""
        if not self.models:
            print("No models trained. Train models first.")
            return
        
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE BY TRIGGER CATEGORY")
        print("=" * 80)
        
        for trigger_cat in self.TRIGGER_CATEGORIES:
            if trigger_cat in self.models:
                importance = self.get_feature_importance(trigger_cat)
                print(f"\n{trigger_cat.upper()}:")
                for idx, (feature, score) in enumerate(list(importance.items())[:top_n]):
                    print(f"  {idx+1:2d}. {feature:20s}: {score:.4f}")
    
    def generate_synthetic_triggers(self, n_samples: int, random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic trigger data based on observed distributions.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
        
        Returns:
            DataFrame with synthetic trigger data
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Analyze existing data if available
        if self.questionnaire_data is not None and len(self.questionnaire_data) > 0:
            # Process existing data to get trigger distributions
            trigger_df = self.process_dataset()
            if len(trigger_df) > 0 and all(cat in trigger_df.columns for cat in self.TRIGGER_CATEGORIES):
                # Use actual distributions from processed data
                return self._generate_from_distributions(trigger_df[self.TRIGGER_CATEGORIES], n_samples, random_state)
        
        # Fallback: Use default distributions based on typical patterns
        return self._generate_from_default_distributions(n_samples, random_state)
    
    def _generate_from_distributions(self, existing_data: pd.DataFrame, n_samples: int, 
                                     random_state: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic data based on existing distributions."""
        if random_state is not None:
            np.random.seed(random_state)
        
        synthetic_data = {}
        
        for col in self.TRIGGER_CATEGORIES:
            if col not in existing_data.columns:
                continue
            
            values = existing_data[col].values
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            min_val = float(np.min(values))
            max_val = float(np.max(values))
            
            # Check if constant
            if std_val == 0 or min_val == max_val:
                synthetic_data[col] = np.full(n_samples, mean_val)
                continue
            
            # Fit beta distribution for bounded 0-1 values
            # Method of moments to estimate alpha and beta
            if mean_val > 0 and mean_val < 1:
                variance = std_val ** 2
                if variance > 0:
                    alpha = mean_val * ((mean_val * (1 - mean_val)) / variance - 1)
                    beta = (1 - mean_val) * ((mean_val * (1 - mean_val)) / variance - 1)
                    
                    # Ensure positive parameters
                    alpha = max(0.1, alpha)
                    beta = max(0.1, beta)
                    
                    # Generate from beta distribution
                    samples = np.random.beta(alpha, beta, size=n_samples)
                    # Clip to original range
                    samples = np.clip(samples, min_val, max_val)
                else:
                    samples = np.full(n_samples, mean_val)
            else:
                # Use truncated normal for edge cases
                samples = np.random.normal(mean_val, std_val, size=n_samples)
                samples = np.clip(samples, min_val, max_val)
            
            synthetic_data[col] = samples
        
        return pd.DataFrame(synthetic_data)
    
    def _generate_from_default_distributions(self, n_samples: int, 
                                            random_state: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic data using default distributions."""
        if random_state is not None:
            np.random.seed(random_state)
        
        synthetic_data = {}
        
        # Stress: Beta distribution (skewed towards lower values)
        synthetic_data['stress'] = np.random.beta(1.5, 2.5, size=n_samples)
        
        # Hormonal: Beta distribution (vary the values, but keep them moderate to high)
        synthetic_data['hormonal'] = np.random.beta(2.0, 1.5, size=n_samples)
        
        # Sleep: Beta distribution (vary the values, but keep them moderate to high)
        synthetic_data['sleep'] = np.random.beta(2.0, 1.5, size=n_samples)
        
        # Weather: Beta distribution (moderate values)
        synthetic_data['weather'] = np.random.beta(2.0, 4.0, size=n_samples)
        
        # Food: Highly skewed towards 1.0 (most values are 1.0)
        # Use mixture: 90% at 1.0, 10% from beta distribution
        food_samples = np.random.beta(10.0, 1.0, size=n_samples)
        # Mix with high probability of 1.0
        mask = np.random.random(n_samples) < 0.9
        food_samples[mask] = 1.0
        synthetic_data['food'] = np.clip(food_samples, 0.27, 1.0)
        
        # Sensory: Beta distribution (moderate to high values)
        synthetic_data['sensory'] = np.random.beta(2.0, 1.5, size=n_samples)
        
        # Physical: Highly skewed towards 0 (most values are low)
        synthetic_data['physical'] = np.random.beta(1.0, 5.0, size=n_samples)
        
        return pd.DataFrame(synthetic_data)


def main():
    """Example usage of ML trigger classifier."""
    import sys
    
    questionnaire_file = 'migraine_symptom_classification.csv'
    
    if len(sys.argv) > 1:
        questionnaire_file = sys.argv[1]
    
    # Initialize classifier
    classifier = MLTriggerClassifier(questionnaire_file)
    
    # Print statistics
    classifier.print_statistics()
    
    # Train models
    classifier.train_models()
    
    # Print feature importance
    classifier.print_feature_importance()
    
    # Process dataset
    result_df = classifier.process_dataset()
    
    if len(result_df) > 0:
        print(f"\nProcessed dataset shape: {result_df.shape}")
        print(f"Trigger columns added: {classifier.TRIGGER_CATEGORIES}")


if __name__ == '__main__':
    main()

