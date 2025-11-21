"""
Main synthetic data generator for migraine prediction.
SOLUTION 3: Separates predictor features from symptom features to eliminate data leakage.

Architecture:
1. Generate PREDICTOR features first (stress, triggers, weather, day factors)
2. Calculate migraine probability from predictors
3. Sample migraine status from probability
4. Generate SYMPTOM features (mood, activity) based on migraine with heavy noise
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import uuid

from .distributions import (
    generate_step_count,
    generate_mood,
    generate_screen_brightness,
    normalize_step_count as normalize_steps,
    get_mood_category
)
from .questionnaire_processor import QuestionnaireProcessor
from .utils import (
    get_weekday_factor,
    get_migraine_activity_factor,
    get_migraine_mood_factor,
    generate_timestamps,
    get_hour_from_timestamp,
    get_day_of_week
)
from .migraine_probability import calculate_migraine_probability, sample_migraine_status
from .distribution_config import DistributionParams, get_train_distribution_params, get_val_test_distribution_params


class SyntheticDataGenerator:
    """Main class for generating synthetic migraine prediction data."""
    
    def __init__(self, person_data_file: Optional[str] = None,
                 questionnaire_file: Optional[str] = None,
                 random_state: Optional[int] = None,
                 distribution_params: Optional[DistributionParams] = None):
        """
        Initialize synthetic data generator.
        
        Args:
            person_data_file: Path to person_data CSV file (optional)
            questionnaire_file: Path to questionnaire CSV file (optional)
            random_state: Random seed for reproducibility
            distribution_params: Distribution parameters to use (default: train params)
        """
        self.random_state = random_state
        self._random_counter = 0  # Counter for unique random states
        if random_state is not None:
            np.random.seed(random_state)
        
        # Set distribution parameters
        self.distribution_params = distribution_params or get_train_distribution_params()
        
        # Load person data if provided
        self.person_data = None
        if person_data_file:
            try:
                self.person_data = pd.read_csv(person_data_file)
                print(f"Loaded person data: {len(self.person_data)} persons")
            except Exception as e:
                print(f"Warning: Could not load person data file {person_data_file}: {e}")
        
        # Initialize questionnaire processor
        self.questionnaire_processor = QuestionnaireProcessor(questionnaire_file)
    
    def generate_person_data(self, n_persons: int) -> pd.DataFrame:
        """
        Generate or load person data.
        
        Args:
            n_persons: Number of persons to generate
        
        Returns:
            DataFrame with person data
        """
        if self.person_data is not None and len(self.person_data) >= n_persons:
            # Use existing person data
            return self.person_data.head(n_persons).copy()
        else:
            # Generate synthetic person data
            persons = []
            for i in range(n_persons):
                person_id = str(uuid.uuid4())
                migraine_days = np.random.randint(1, 15)  # 1-14 migraine days per month
                
                # Generate trigger profile
                trigger_profile = self.questionnaire_processor.generate_user_trigger_profile()
                
                person = {
                    'person_id': person_id,
                    'migraine_days_per_month': migraine_days,
                    'trigger_stress': trigger_profile.get('stress', 0.0),
                    'trigger_hormonal': trigger_profile.get('hormonal', 0.0),
                    'trigger_sleep': trigger_profile.get('sleep', 0.0),
                    'trigger_weather': trigger_profile.get('weather', 0.0),
                    'trigger_food': trigger_profile.get('food', 0.0),
                    'trigger_sensory': trigger_profile.get('sensory', 0.0),
                    'trigger_physical': trigger_profile.get('physical', 0.0)
                }
                persons.append(person)
            
            return pd.DataFrame(persons)
    
    def generate_daily_data(self, person_row: pd.Series, n_days: int,
                           start_date: datetime, predictor_features: Optional[pd.DataFrame] = None,
                           distribution_params: Optional[DistributionParams] = None) -> pd.DataFrame:
        """
        Generate daily data for a single person using Solution 3 architecture.
        
        SOLUTION 3 FLOW:
        1. Generate PREDICTOR features first (stress, triggers, weather, day factors)
        2. Calculate migraine probability from predictors
        3. Sample migraine status from probability
        4. Generate SYMPTOM features (mood, activity) based on migraine with heavy noise
        
        Args:
            person_row: Row from person data DataFrame
            n_days: Number of days to generate
            start_date: Starting date
            predictor_features: Optional DataFrame with predictor features (stress, triggers, weather)
            distribution_params: Distribution parameters to use (default: instance params)
        
        Returns:
            DataFrame with daily data
        """
        # Use provided params or instance params
        params = distribution_params or self.distribution_params
        
        person_id = person_row['person_id']
        
        # Get person's trigger sensitivities (predictor features)
        trigger_stress = person_row.get('trigger_stress', 0.5)
        trigger_hormonal = person_row.get('trigger_hormonal', 0.5)
        trigger_sleep = person_row.get('trigger_sleep', 0.5)
        trigger_weather = person_row.get('trigger_weather', 0.5)
        trigger_food = person_row.get('trigger_food', 0.5)
        trigger_sensory = person_row.get('trigger_sensory', 0.5)
        trigger_physical = person_row.get('trigger_physical', 0.5)
        
        # Generate timestamps
        timestamps = generate_timestamps(start_date, n_days, hour_variation=True)
        
        # PHASE 1: Initialize temporal tracking variables
        consecutive_migraine_days = 0
        days_since_last_migraine = 0
        
        # Initialize data lists
        data_rows = []
        
        for day_idx, timestamp in enumerate(timestamps):
            # ============================================================
            # STEP 1: Generate PREDICTOR features FIRST (independent of migraine)
            # ============================================================
            day_of_week = get_day_of_week(timestamp)
            hour_of_day = get_hour_from_timestamp(timestamp)
            
            # Generate predictor features independently
            # These should be available BEFORE migraine occurs
            if predictor_features is not None and day_idx < len(predictor_features):
                # Use provided predictor features (from combined_data)
                stress_intensity = predictor_features.iloc[day_idx].get('stress_intensity', np.random.randint(0, 10))
                weather_pressure = predictor_features.iloc[day_idx].get('pressure_mean', np.random.normal(1012, 5))
                weather_temp = predictor_features.iloc[day_idx].get('temp_mean', np.random.normal(5, 2))
                # Get trigger values from predictor features
                daily_stress = predictor_features.iloc[day_idx].get('stress', np.random.beta(1.5, 2.5))
                daily_hormonal = predictor_features.iloc[day_idx].get('hormonal', np.random.beta(2.0, 1.5))
                daily_sleep = predictor_features.iloc[day_idx].get('sleep', np.random.beta(2.0, 1.5))
                daily_weather = predictor_features.iloc[day_idx].get('weather', np.random.beta(2.0, 4.0))
                daily_food = predictor_features.iloc[day_idx].get('food', np.random.beta(10.0, 1.0))
                daily_sensory = predictor_features.iloc[day_idx].get('sensory', np.random.beta(2.0, 1.5))
                daily_physical = predictor_features.iloc[day_idx].get('physical', np.random.beta(1.0, 5.0))
            else:
                # Generate predictor features independently using distribution params
                stress_intensity = np.random.randint(0, params.stress_intensity_max + 1)
                weather_pressure = np.random.normal(params.pressure_mean, params.pressure_std)
                weather_temp = np.random.normal(params.temp_mean, params.temp_std)
                # Daily trigger levels (can vary day-to-day) - use params for stress
                daily_stress = np.random.beta(params.trigger_beta_alpha, params.trigger_beta_beta)
                daily_hormonal = np.random.beta(2.0, 1.5)
                daily_sleep = np.random.beta(2.0, 1.5)
                daily_weather = np.random.beta(2.0, 4.0)
                daily_food = np.random.beta(10.0, 1.0)
                daily_sensory = np.random.beta(2.0, 1.5)
                daily_physical = np.random.beta(1.0, 5.0)
            
            # ============================================================
            # STEP 2: Calculate migraine probability from PREDICTORS
            # ============================================================
            # Use predictor features to calculate probability
            # Combine person's sensitivity with daily trigger levels
            # PHASE 1: Include temporal momentum factors (use PREVIOUS day's values)
            migraine_prob = calculate_migraine_probability(
                stress_intensity=stress_intensity,
                trigger_stress=trigger_stress * daily_stress,  # Person sensitivity × daily level
                trigger_hormonal=trigger_hormonal * daily_hormonal,
                trigger_sleep=trigger_sleep * daily_sleep,
                trigger_weather=trigger_weather * daily_weather,
                trigger_food=trigger_food * daily_food,
                trigger_sensory=trigger_sensory * daily_sensory,
                trigger_physical=trigger_physical * daily_physical,
                weather_pressure=weather_pressure,
                weather_temp=weather_temp,
                day_of_week=day_of_week,
                consecutive_migraine_days=consecutive_migraine_days,  # PHASE 1: Use previous day's value
                days_since_last_migraine=days_since_last_migraine,  # PHASE 1: Use previous day's value
                base_probability=params.base_migraine_probability,  # Use distribution params
                random_state=None  # Don't reset seed - use global random state
            )
            
            # ============================================================
            # STEP 3: Sample migraine status from probability
            # ============================================================
            has_migraine = sample_migraine_status(migraine_prob, random_state=None)  # Use global random state
            
            # PHASE 1: Update temporal tracking variables AFTER determining today's migraine
            # (These will be used for the NEXT day's probability calculation)
            if has_migraine:
                consecutive_migraine_days += 1
                days_since_last_migraine = 0
            else:
                consecutive_migraine_days = 0
                days_since_last_migraine += 1
            
            # ============================================================
            # STEP 4: Generate SYMPTOM features (mood, activity) AFTER migraine
            # ============================================================
            # These are consequences of migraine, but with HEAVY noise to prevent perfect prediction
            weekday_factor = get_weekday_factor(day_of_week)
            migraine_activity_factor = get_migraine_activity_factor(has_migraine, params=params)
            migraine_mood_factor = get_migraine_mood_factor(has_migraine, params=params)
            
            # Generate step count (symptom - affected by migraine but with noise)
            # CRITICAL FIX: Use random_state=None to use global random state (not fixed seed)
            # This prevents deterministic behavior where same seed always produces same result
            step_count = generate_step_count(
                n_samples=1,
                weekday_factor=weekday_factor,
                migraine_factor=migraine_activity_factor,
                random_state=None,  # Use global random state (already seeded in __init__)
                params=params
            )[0]
            
            # Generate mood (symptom - affected by migraine but with heavy noise)
            # CRITICAL FIX: Use random_state=None to use global random state (not fixed seed)
            # This prevents deterministic behavior where same seed always produces same result
            mood_score = generate_mood(
                n_samples=1,
                day_of_week=day_of_week,
                migraine_factor=migraine_mood_factor,
                random_state=None,  # Use global random state (already seeded in __init__)
                params=params
            )[0]
            mood_category = get_mood_category(mood_score)
            
            # Generate screen brightness (independent - not a symptom)
            screen_brightness = generate_screen_brightness(
                n_samples=1,
                hour_of_day=hour_of_day,
                random_state=self.random_state
            )[0]
            
            # Get additional weather features (handle case where predictor_features might not have all columns)
            if predictor_features is not None and day_idx < len(predictor_features):
                wind_mean = predictor_features.iloc[day_idx].get('wind_mean', np.random.gamma(2.0, 1.5))
                sun_irr_mean = predictor_features.iloc[day_idx].get('sun_irr_mean', np.random.gamma(3.0, 30.0))
                sun_time_mean = predictor_features.iloc[day_idx].get('sun_time_mean', np.random.exponential(0.5))
                precip_total = predictor_features.iloc[day_idx].get('precip_total', np.random.exponential(0.3))
                cloud_mean = predictor_features.iloc[day_idx].get('cloud_mean', np.random.normal(50.0, 30.0))
            else:
                wind_mean = np.random.gamma(2.0, 1.5)
                sun_irr_mean = np.random.gamma(3.0, 30.0)
                sun_time_mean = np.random.exponential(0.5)
                precip_total = np.random.exponential(0.3)
                cloud_mean = np.random.normal(50.0, 30.0)
            
            # Create data row with both predictor and symptom features
            row = {
                # PREDICTOR features (cause migraines)
                'stress_intensity': int(stress_intensity),
                'temp_mean': float(weather_temp),
                'wind_mean': float(wind_mean),
                'pressure_mean': float(weather_pressure),
                'sun_irr_mean': float(sun_irr_mean),
                'sun_time_mean': float(sun_time_mean),
                'precip_total': float(precip_total),
                'cloud_mean': float(cloud_mean),
                'stress': float(daily_stress),
                'hormonal': float(daily_hormonal),
                'sleep': float(daily_sleep),
                'weather': float(daily_weather),
                'food': float(daily_food),
                'sensory': float(daily_sensory),
                'physical': float(daily_physical),
                # PHASE 1: Temporal features
                'consecutive_migraine_days': int(consecutive_migraine_days),
                'days_since_last_migraine': int(days_since_last_migraine),
                # SYMPTOM features (result from migraines)
                'step_count': int(step_count),
                'mood_score': float(mood_score),
                'mood_category': mood_category,
                'screen_brightness': float(screen_brightness),
                # TARGET
                'migraine': bool(has_migraine)
            }
            
            data_rows.append(row)
        
        return pd.DataFrame(data_rows)
    
    def _generate_predictor_features(self, n_days: int, random_state: Optional[int] = None,
                                     distribution_params: Optional[DistributionParams] = None) -> pd.DataFrame:
        """
        Generate predictor features independently (before determining migraine).
        These features should CAUSE migraines, not be caused by them.
        
        Args:
            n_days: Number of days to generate features for
            random_state: Random seed
            distribution_params: Distribution parameters to use (default: instance params)
        
        Returns:
            DataFrame with predictor features (stress, triggers, weather)
        """
        # Use provided params or instance params
        params = distribution_params or self.distribution_params
        
        if random_state is not None:
            np.random.seed(random_state)
        
        predictor_data = {
            # Stress intensity (0-max scale)
            'stress_intensity': np.random.randint(0, params.stress_intensity_max + 1, size=n_days),
            
            # Daily trigger levels (can vary day-to-day) - use params for stress trigger
            'stress': np.random.beta(params.trigger_beta_alpha, params.trigger_beta_beta, size=n_days),
            'hormonal': np.random.beta(2.0, 1.5, size=n_days),
            'sleep': np.random.beta(2.0, 1.5, size=n_days),
            'weather': np.random.beta(2.0, 4.0, size=n_days),
            'food': np.random.beta(10.0, 1.0, size=n_days),
            'sensory': np.random.beta(2.0, 1.5, size=n_days),
            'physical': np.random.beta(1.0, 5.0, size=n_days),
            
            # Weather features - use params
            'temp_mean': np.random.normal(params.temp_mean, params.temp_std, size=n_days).clip(0, 15),
            'wind_mean': np.random.gamma(2.0, 1.5, size=n_days).clip(0, 10),
            'pressure_mean': np.random.normal(params.pressure_mean, params.pressure_std, size=n_days).clip(1000, 1025),
            'sun_irr_mean': np.random.gamma(3.0, 30.0, size=n_days).clip(0, 200),
            'sun_time_mean': np.random.exponential(0.5, size=n_days).clip(0, 5),
            'precip_total': np.random.exponential(0.3, size=n_days).clip(0, 5),
            'cloud_mean': np.random.normal(50.0, 30.0, size=n_days).clip(0, 100),
        }
        
        return pd.DataFrame(predictor_data)
    
    def generate_dataset(self, n_persons: int, n_days: int,
                        start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate complete dataset using Solution 3 architecture.
        
        SOLUTION 3 FLOW:
        1. Generate predictor features for each person independently
        2. Use predictors to determine migraine probabilistically
        3. Generate symptom features based on migraine with heavy noise
        
        Args:
            n_persons: Number of persons
            n_days: Number of days per person
            start_date: Starting date (default: today)
        
        Returns:
            DataFrame with complete dataset
        """
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate or load person data
        person_data = self.generate_person_data(n_persons)
        
        # Generate daily data for each person
        all_data = []
        
        for idx, person_row in person_data.iterrows():
            if idx % 100 == 0:
                print(f"Generating data for person {idx + 1}/{n_persons}...")
            
            # Offset start date for each person to create variety
            person_start_date = start_date + timedelta(days=np.random.randint(0, 30))
            
            # Generate predictor features for this person (independent of migraine)
            # Use instance distribution params
            predictor_features = self._generate_predictor_features(
                n_days, 
                random_state=self.random_state,
                distribution_params=self.distribution_params
            )
            
            # Generate daily data using Solution 3 (predictors → migraine → symptoms)
            # Use instance distribution params
            daily_data = self.generate_daily_data(
                person_row, 
                n_days, 
                person_start_date,
                predictor_features=predictor_features,
                distribution_params=self.distribution_params
            )
            all_data.append(daily_data)
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Normalize step_count across entire dataset
        if 'step_count' in combined_data.columns:
            step_counts = combined_data['step_count'].values
            step_count_normalized = normalize_steps(step_counts, method='log')
            combined_data['step_count_normalized'] = step_count_normalized
            # Remove original step_count column
            combined_data = combined_data.drop(columns=['step_count'])
        
        # Normalize screen_brightness across entire dataset
        if 'screen_brightness' in combined_data.columns:
            brightness_values = combined_data['screen_brightness'].values
            # Min-max normalization for brightness
            brightness_min = brightness_values.min()
            brightness_max = brightness_values.max()
            if brightness_max > brightness_min:
                brightness_normalized = (brightness_values - brightness_min) / (brightness_max - brightness_min)
            else:
                brightness_normalized = np.zeros_like(brightness_values)
            combined_data['screen_brightness_normalized'] = brightness_normalized
            # Remove original screen_brightness column
            combined_data = combined_data.drop(columns=['screen_brightness'])
        
        # Normalize step_count and screen_brightness (symptom features)
        # Note: Predictor features are kept as-is (not normalized here)
        if 'step_count' in combined_data.columns:
            step_counts = combined_data['step_count'].values
            step_count_normalized = normalize_steps(step_counts, method='log')
            combined_data['step_count_normalized'] = step_count_normalized
            combined_data = combined_data.drop(columns=['step_count'])
        
        if 'screen_brightness' in combined_data.columns:
            brightness_values = combined_data['screen_brightness'].values
            brightness_min = brightness_values.min()
            brightness_max = brightness_values.max()
            if brightness_max > brightness_min:
                brightness_normalized = (brightness_values - brightness_min) / (brightness_max - brightness_min)
            else:
                brightness_normalized = np.zeros_like(brightness_values)
            combined_data['screen_brightness_normalized'] = brightness_normalized
            combined_data = combined_data.drop(columns=['screen_brightness'])
        
        # Reorder columns: predictors first, then temporal, then symptoms, then target
        predictor_cols = ['stress_intensity', 'temp_mean', 'wind_mean', 'pressure_mean', 
                         'sun_irr_mean', 'sun_time_mean', 'precip_total', 'cloud_mean',
                         'stress', 'hormonal', 'sleep', 'weather', 'food', 'sensory', 'physical']
        temporal_cols = ['consecutive_migraine_days', 'days_since_last_migraine']  # PHASE 1: Temporal features
        symptom_cols = ['step_count_normalized', 'mood_score', 'mood_category', 'screen_brightness_normalized']
        target_cols = ['migraine']
        
        # Get available columns
        available_predictors = [col for col in predictor_cols if col in combined_data.columns]
        available_temporal = [col for col in temporal_cols if col in combined_data.columns]  # PHASE 1
        available_symptoms = [col for col in symptom_cols if col in combined_data.columns]
        available_targets = [col for col in target_cols if col in combined_data.columns]
        
        # Reorder: predictors, temporal, symptoms, target
        ordered_cols = available_predictors + available_temporal + available_symptoms + available_targets
        combined_data = combined_data[ordered_cols]
        
        print(f"Generated dataset: {len(combined_data)} records")
        print(f"  Predictor features: {len(available_predictors)}")
        print(f"  Temporal features: {len(available_temporal)}")  # PHASE 1
        print(f"  Symptom features: {len(available_symptoms)}")
        return combined_data
    
    def save_dataset(self, dataset: pd.DataFrame, output_file: str):
        """
        Save dataset to CSV file.
        
        Args:
            dataset: DataFrame to save
            output_file: Output file path
        """
        # Save directly (no timestamp conversion needed)
        dataset.to_csv(output_file, index=False)
        print(f"Saved dataset to {output_file}")

