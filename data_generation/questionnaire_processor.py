"""
Questionnaire data processor for analyzing and classifying migraine triggers.
Processes migraine symptom classification data to extract trigger patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class QuestionnaireProcessor:
    """Processes questionnaire data to classify triggers into categories."""
    
    # Trigger categories
    TRIGGER_CATEGORIES = [
        'stress',
        'hormonal',
        'sleep',
        'weather',
        'food',
        'sensory',
        'physical'
    ]
    
    def __init__(self, questionnaire_file: str = None, use_ml: bool = False):
        """
        Initialize questionnaire processor.
        
        Args:
            questionnaire_file: Path to questionnaire CSV file (optional)
            use_ml: Whether to use ML classifier (requires ml_trigger_classifier)
        """
        self.questionnaire_data = None
        self.ml_classifier = None
        self.use_ml = use_ml
        
        if questionnaire_file:
            self.load_questionnaire(questionnaire_file)
        
        if use_ml and questionnaire_file:
            try:
                from .ml_trigger_classifier import MLTriggerClassifier
                self.ml_classifier = MLTriggerClassifier(questionnaire_file)
                self.ml_classifier.train_models()
                print("ML classifier initialized and trained.")
            except ImportError:
                print("Warning: ML classifier not available. Using rule-based classification.")
                self.use_ml = False
            except Exception as e:
                print(f"Warning: Could not initialize ML classifier: {e}. Using rule-based classification.")
                self.use_ml = False
    
    def load_questionnaire(self, file_path: str):
        """
        Load questionnaire data from CSV file.
        
        Args:
            file_path: Path to CSV file
        """
        try:
            self.questionnaire_data = pd.read_csv(file_path)
            print(f"Loaded questionnaire data: {len(self.questionnaire_data)} records")
        except Exception as e:
            print(f"Warning: Could not load questionnaire file {file_path}: {e}")
            self.questionnaire_data = None
    
    def classify_triggers_from_symptoms(self, row: pd.Series) -> Dict[str, float]:
        """
        Classify triggers from symptom data.
        
        Uses ML classifier if available, otherwise uses rule-based approach.
        
        Maps symptoms to trigger categories:
        - Stress: High intensity, frequency, duration
        - Hormonal: Related to menstruation patterns (if applicable)
        - Sleep: Sleep-related symptoms
        - Weather: Environmental factors
        - Food: Nausea, vomiting (digestive symptoms)
        - Sensory: Photophobia, Phonophobia, Visual, Sensory symptoms
        - Physical: Physical activity, vertigo, ataxia
        
        Args:
            row: Row from questionnaire data
        
        Returns:
            Dictionary of trigger severity scores (0-1) for each category
        """
        # Use ML classifier if available
        if self.use_ml and self.ml_classifier is not None:
            return self.ml_classifier.classify_triggers(row)
        
        # Fallback to rule-based
        triggers = {cat: 0.0 for cat in self.TRIGGER_CATEGORIES}
        
        # Stress trigger: Based on intensity, frequency, duration
        intensity = row.get('Intensity', 0)
        frequency = row.get('Frequency', 0)
        duration = row.get('Duration', 0)
        stress_score = (intensity + frequency + duration) / 15.0  # Normalize to 0-1
        triggers['stress'] = min(stress_score, 1.0)
        
        # Hormonal trigger: Based on age and gender patterns (if data available)
        # This would typically come from person_data, but we can infer from symptoms
        age = row.get('Age', 30)
        # Higher hormonal sensitivity in certain age ranges
        if 20 <= age <= 50:
            triggers['hormonal'] = 0.3 + (intensity / 10.0) * 0.4
        else:
            triggers['hormonal'] = 0.1 + (intensity / 10.0) * 0.2
        triggers['hormonal'] = min(triggers['hormonal'], 1.0)
        
        # Sleep trigger: Can be inferred from duration and other factors
        # In real data, this would come from sleep_duration, sleep_deficit
        triggers['sleep'] = 0.2 + (duration / 5.0) * 0.3
        triggers['sleep'] = min(triggers['sleep'], 1.0)
        
        # Weather trigger: Hard to infer from symptoms alone
        # Would typically come from weather data correlation
        triggers['weather'] = 0.1 + np.random.beta(1.5, 2.5) * 0.3
        
        # Food trigger: Based on nausea and vomiting
        nausea = row.get('Nausea', 0)
        vomit = row.get('Vomit', 0)
        food_score = (nausea + vomit) / 2.0
        triggers['food'] = min(food_score, 1.0)
        
        # Sensory trigger: Based on photophobia, phonophobia, visual, sensory symptoms
        photophobia = row.get('Photophobia', 0)
        phonophobia = row.get('Phonophobia', 0)
        visual = row.get('Visual', 0)
        sensory = row.get('Sensory', 0)
        sensory_score = (photophobia + phonophobia + visual + sensory) / 8.0
        triggers['sensory'] = min(sensory_score, 1.0)
        
        # Physical trigger: Based on vertigo, ataxia, physical symptoms
        vertigo = row.get('Vertigo', 0)
        ataxia = row.get('Ataxia', 0)
        physical_score = (vertigo + ataxia) / 2.0
        triggers['physical'] = min(physical_score, 1.0)
        
        return triggers
    
    def process_questionnaire_data(self) -> pd.DataFrame:
        """
        Process questionnaire data to extract trigger patterns.
        
        Returns:
            DataFrame with trigger classifications
        """
        if self.questionnaire_data is None:
            print("No questionnaire data loaded. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Classify triggers for each row
        trigger_data = []
        for idx, row in self.questionnaire_data.iterrows():
            triggers = self.classify_triggers_from_symptoms(row)
            trigger_data.append(triggers)
        
        trigger_df = pd.DataFrame(trigger_data)
        
        # Add original columns for reference
        for col in ['Age', 'Intensity', 'Frequency', 'Duration', 'Type']:
            if col in self.questionnaire_data.columns:
                trigger_df[col] = self.questionnaire_data[col].values
        
        return trigger_df
    
    def generate_user_trigger_profile(self, person_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Generate a user-specific trigger profile.
        
        If person_data is provided, use existing trigger values.
        Otherwise, generate synthetic trigger profile based on learned patterns.
        
        Args:
            person_data: DataFrame with person data including trigger columns
        
        Returns:
            Dictionary of trigger severity scores (0-1) for each category
        """
        if person_data is not None and len(person_data) > 0:
            # Use existing trigger data from person_data
            row = person_data.iloc[0]
            triggers = {}
            
            # Map existing trigger columns
            trigger_mapping = {
                'trigger_stress': 'stress',
                'trigger_hormones': 'hormonal',
                'trigger_sleep': 'sleep',
                'trigger_weather': 'weather',
                'trigger_meals': 'food'
            }
            
            for col, category in trigger_mapping.items():
                if col in row:
                    triggers[category] = float(row[col])
            
            # Add missing categories with default values
            for cat in self.TRIGGER_CATEGORIES:
                if cat not in triggers:
                    triggers[cat] = np.random.beta(1.5, 2.5)
            
            return triggers
        else:
            # Generate synthetic trigger profile
            triggers = {}
            for cat in self.TRIGGER_CATEGORIES:
                triggers[cat] = np.random.beta(1.5, 2.5)
            
            return triggers
    
    def get_trigger_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about trigger patterns from questionnaire data.
        
        Returns:
            Dictionary with statistics for each trigger category
        """
        if self.questionnaire_data is None:
            return {}
        
        processed = self.process_questionnaire_data()
        
        stats = {}
        for cat in self.TRIGGER_CATEGORIES:
            if cat in processed.columns:
                stats[cat] = {
                    'mean': float(processed[cat].mean()),
                    'std': float(processed[cat].std()),
                    'min': float(processed[cat].min()),
                    'max': float(processed[cat].max()),
                    'median': float(processed[cat].median())
                }
        
        return stats

