"""
Distribution configuration for train/val/test split with distribution shift.
Allows different distribution parameters for training vs validation/test sets
to better detect overfitting.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DistributionParams:
    """Parameters for data generation distributions."""
    
    # Mood generation parameters
    # CRITICAL FIX: Reduced noise scale from 7.5 to 1.5 (was causing all values to clip to 0)
    # Mood range is 0-10, so noise scale of 7.5 was way too large (could be ±22.5!)
    mood_noise_scale: float = 1.5  # Noise scale for mood generation (reduced from 7.5)
    mood_uniform_noise_range: float = 0.5  # Uniform noise range for mood (reduced from 2.5)
    
    # Activity/mood factor ranges
    # CRITICAL FIX: Increase minimums to prevent all zeros
    # The issue: mood_factor=0.1 * base_mood can result in very low values that get clipped to 0
    # Solution: Use higher minimums and ensure noise is applied BEFORE clipping
    migraine_mood_factor_min: float = 0.4  # CRITICAL: Increased from 0.1 to prevent zeros
    migraine_mood_factor_max: float = 1.0
    non_migraine_mood_factor_lower_min: float = 0.4  # CRITICAL: Increased from 0.2
    non_migraine_mood_factor_lower_max: float = 0.9
    non_migraine_mood_factor_upper_min: float = 0.5
    non_migraine_mood_factor_upper_max: float = 1.0
    non_migraine_mood_lower_chance: float = 0.4
    
    # CRITICAL FIX: Increase minimums to prevent perfect separation
    migraine_activity_factor_min: float = 0.3  # CRITICAL: Increased from 0.2
    migraine_activity_factor_max: float = 1.0
    non_migraine_activity_factor_lower_min: float = 0.4  # CRITICAL: Increased from 0.3
    non_migraine_activity_factor_lower_max: float = 0.9
    non_migraine_activity_factor_upper_min: float = 0.5
    non_migraine_activity_factor_upper_max: float = 1.0
    non_migraine_activity_lower_chance: float = 0.4
    
    # Step count noise
    step_count_noise_scale: float = 0.13  # 15% of mean as noise
    
    # Predictor feature distributions
    stress_intensity_max: int = 10
    trigger_beta_alpha: float = 1.5  # For stress trigger
    trigger_beta_beta: float = 2.5
    
    # Weather distributions
    temp_mean: float = 5.0
    temp_std: float = 2.0
    pressure_mean: float = 1012.0
    pressure_std: float = 5.0
    
    # Base migraine probability
    base_migraine_probability: float = 0.08
    
    # Temporal momentum
    momentum_boost_per_day: float = 0.10  # +10% per consecutive day
    momentum_max_boost: float = 0.30  # Capped at 30%


def get_train_distribution_params() -> DistributionParams:
    """
    Get distribution parameters for training data.
    These are the "standard" parameters.
    """
    return DistributionParams(
        mood_noise_scale=1.5,  # CRITICAL FIX: Reduced from 7.5 to prevent clipping to 0
        mood_uniform_noise_range=0.5,  # CRITICAL FIX: Reduced from 2.5
        step_count_noise_scale=0.15,
        base_migraine_probability=0.08,
        # CRITICAL FIX: Use higher minimums to prevent perfect separation (all zeros)
        migraine_mood_factor_min=0.3,  # Increased from 0.1 to prevent zeros
        migraine_mood_factor_max=1.0,
        non_migraine_mood_factor_lower_min=0.4,  # Increased from 0.2
        non_migraine_mood_factor_lower_max=0.9,
        non_migraine_mood_factor_upper_min=0.5,
        non_migraine_mood_factor_upper_max=1.0,
        non_migraine_mood_lower_chance=0.4,
        migraine_activity_factor_min=0.3,  # Increased from 0.2
        migraine_activity_factor_max=1.0,
        non_migraine_activity_factor_lower_min=0.4,  # Increased from 0.3
        non_migraine_activity_factor_lower_max=0.9,
        non_migraine_activity_factor_upper_min=0.5,
        non_migraine_activity_factor_upper_max=1.0,
        non_migraine_activity_lower_chance=0.4
    )


def get_val_test_distribution_params() -> DistributionParams:
    """
    Get distribution parameters for validation/test data.
    These are slightly different to test generalization.
    
    Changes:
    - Slightly higher noise (more realistic variability)
    - Slightly different weather patterns
    - Slightly different trigger distributions
    - Slightly different base probability
    """
    return DistributionParams(
        # Slightly higher noise to test robustness (but still reasonable)
        mood_noise_scale=1.65,  # Slightly higher than train (1.5) but much lower than old 8.0
        mood_uniform_noise_range=0.6,  # Slightly higher than train (0.5) but much lower than old 3.0
        step_count_noise_scale=0.165,  # Increased from 0.15
        
        # Slightly different mood/activity ranges (more overlap)
        # CRITICAL FIX: Use higher minimums to prevent perfect separation
        migraine_mood_factor_min=0.35,  # Increased from 0.15
        migraine_mood_factor_max=1.0,
        non_migraine_mood_factor_lower_min=0.45,  # Increased from 0.25
        non_migraine_mood_factor_lower_max=0.95,
        non_migraine_mood_factor_upper_min=0.55,
        non_migraine_mood_factor_upper_max=1.0,
        non_migraine_mood_lower_chance=0.4,
        
        migraine_activity_factor_min=0.30,  # Increased from 0.25
        migraine_activity_factor_max=1.0,
        non_migraine_activity_factor_lower_min=0.45,  # Increased from 0.35
        non_migraine_activity_factor_lower_max=0.95,
        non_migraine_activity_factor_upper_min=0.55,
        non_migraine_activity_factor_upper_max=1.0,
        non_migraine_activity_lower_chance=0.4,
        
        # Slightly different weather patterns
        temp_mean=6.0,  # Slightly warmer
        temp_std=2.5,  # Slightly more variable
        pressure_mean=1013.0,  # Slightly higher pressure
        pressure_std=6.0,  # Slightly more variable
        
        # Slightly different trigger distributions
        trigger_beta_alpha=1.6,  # Slightly different shape
        trigger_beta_beta=2.4,
        
        # Slightly different base probability
        base_migraine_probability=0.09,  # Slightly higher
        
        # Same momentum (this is a learned pattern, should be consistent)
        momentum_boost_per_day=0.10,
        momentum_max_boost=0.30
    )

