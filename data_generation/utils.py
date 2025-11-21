"""
Utility functions for synthetic data generation.
Includes normalization, correlation modeling, and helper functions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


def normalize_step_count(step_counts: np.ndarray, method: str = 'log') -> np.ndarray:
    """
    Normalize step count values to prevent extreme outliers from skewing the model.
    
    Args:
        step_counts: Array of step counts
        method: Normalization method ('log' or 'minmax')
    
    Returns:
        Normalized step counts (0-1 range)
    """
    step_counts = np.array(step_counts, dtype=float)
    
    if method == 'log':
        # Log transform with offset to handle zeros
        log_steps = np.log1p(step_counts)  # log(1 + x) to handle zeros
        # Min-max normalize to 0-1
        if log_steps.max() > log_steps.min():
            normalized = (log_steps - log_steps.min()) / (log_steps.max() - log_steps.min())
        else:
            normalized = np.zeros_like(log_steps)
    elif method == 'minmax':
        # Min-max normalization
        if step_counts.max() > step_counts.min():
            normalized = (step_counts - step_counts.min()) / (step_counts.max() - step_counts.min())
        else:
            normalized = np.zeros_like(step_counts)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def get_weekday_factor(day_of_week: int) -> float:
    """
    Get weekday factor for step count (lower on weekends).
    
    Args:
        day_of_week: Day of week (0=Monday, 6=Sunday)
    
    Returns:
        Multiplier factor (0.7-1.0)
    """
    # Weekends (Saturday=5, Sunday=6) have lower activity
    if day_of_week >= 5:  # Saturday or Sunday
        return 0.75
    else:
        return 1.0


def get_migraine_activity_factor(has_migraine: bool,
                                 params=None,
                                 migraine_min: float = None,
                                 migraine_max: float = None,
                                 non_migraine_min: float = None,
                                 non_migraine_max: float = None,
                                 non_migraine_lower_chance: float = None) -> float:
    """
    Get activity factor for migraine days (nearly identical ranges for maximum overlap).
    PHASE 0: Make ranges nearly identical to break symptom feature separability.
    
    Args:
        has_migraine: Whether person has migraine
        params: DistributionParams object (if None, uses defaults)
        migraine_min: Minimum factor for migraine days (overrides params)
        migraine_max: Maximum factor for migraine days (overrides params)
        non_migraine_min: Minimum factor for non-migraine days (overrides params)
        non_migraine_max: Maximum factor for non-migraine days (overrides params)
        non_migraine_lower_chance: Chance of lower activity for non-migraine (overrides params)
    
    Returns:
        Multiplier factor with substantial overlap
    """
    # Use params if provided, otherwise use function arguments or defaults
    if params is not None:
        migraine_min_val = migraine_min if migraine_min is not None else params.migraine_activity_factor_min
        migraine_max_val = migraine_max if migraine_max is not None else params.migraine_activity_factor_max
        non_migraine_min_val = non_migraine_min if non_migraine_min is not None else params.non_migraine_activity_factor_lower_min
        # BUG FIX: Separate lower_max and upper_max for non-migraine ranges
        non_migraine_lower_max_val = params.non_migraine_activity_factor_lower_max
        non_migraine_upper_max_val = non_migraine_max if non_migraine_max is not None else params.non_migraine_activity_factor_upper_max
        lower_chance = non_migraine_lower_chance if non_migraine_lower_chance is not None else params.non_migraine_activity_lower_chance
    else:
        # Use function arguments or defaults (CRITICAL: Use higher defaults to prevent zeros)
        migraine_min_val = migraine_min if migraine_min is not None else 0.3  # Increased from 0.2
        migraine_max_val = migraine_max if migraine_max is not None else 1.0
        non_migraine_min_val = non_migraine_min if non_migraine_min is not None else 0.4  # Increased from 0.3
        non_migraine_lower_max_val = 0.9  # Default lower max
        non_migraine_upper_max_val = non_migraine_max if non_migraine_max is not None else 1.0
        lower_chance = non_migraine_lower_chance if non_migraine_lower_chance is not None else 0.4
    
    if has_migraine:
        # Migraine days can have any activity level in specified range
        return np.random.uniform(migraine_min_val, migraine_max_val)
    else:
        # Non-migraine days: create substantial overlap
        if np.random.random() < lower_chance:
            # Use lower range (but still high enough to overlap with migraine)
            return np.random.uniform(non_migraine_min_val, non_migraine_lower_max_val)
        else:
            # Use upper range (normal/high activity)
            return np.random.uniform(0.5, non_migraine_upper_max_val)


def get_migraine_mood_factor(has_migraine: bool,
                            params=None,
                            migraine_min: float = None,
                            migraine_max: float = None,
                            non_migraine_min: float = None,
                            non_migraine_max: float = None,
                            non_migraine_lower_chance: float = None) -> float:
    """
    Get mood factor for migraine days (nearly identical ranges for maximum overlap).
    PHASE 0: Make ranges nearly identical to break symptom feature separability.
    
    Args:
        has_migraine: Whether person has migraine
        params: DistributionParams object (if None, uses defaults)
        migraine_min: Minimum factor for migraine days (overrides params)
        migraine_max: Maximum factor for migraine days (overrides params)
        non_migraine_min: Minimum factor for non-migraine days (overrides params)
        non_migraine_max: Maximum factor for non-migraine days (overrides params)
        non_migraine_lower_chance: Chance of lower mood for non-migraine (overrides params)
    
    Returns:
        Multiplier factor with substantial overlap
    """
    # Use params if provided, otherwise use function arguments or defaults
    if params is not None:
        migraine_min_val = migraine_min if migraine_min is not None else params.migraine_mood_factor_min
        migraine_max_val = migraine_max if migraine_max is not None else params.migraine_mood_factor_max
        non_migraine_min_val = non_migraine_min if non_migraine_min is not None else params.non_migraine_mood_factor_lower_min
        # BUG FIX: Separate lower_max and upper_max for non-migraine ranges
        non_migraine_lower_max_val = params.non_migraine_mood_factor_lower_max
        non_migraine_upper_max_val = non_migraine_max if non_migraine_max is not None else params.non_migraine_mood_factor_upper_max
        lower_chance = non_migraine_lower_chance if non_migraine_lower_chance is not None else params.non_migraine_mood_lower_chance
    else:
        # Use function arguments or defaults (CRITICAL: Use higher defaults to prevent zeros)
        migraine_min_val = migraine_min if migraine_min is not None else 0.3  # Increased from 0.1 to prevent zeros
        migraine_max_val = migraine_max if migraine_max is not None else 1.0
        non_migraine_min_val = non_migraine_min if non_migraine_min is not None else 0.4  # Increased from 0.2
        non_migraine_lower_max_val = 0.9  # Default lower max
        non_migraine_upper_max_val = non_migraine_max if non_migraine_max is not None else 1.0
        lower_chance = non_migraine_lower_chance if non_migraine_lower_chance is not None else 0.4
    
    if has_migraine:
        # Migraine days can have any mood level in specified range
        # CRITICAL: Ensure minimum is high enough to prevent zeros after noise/clipping
        return np.random.uniform(migraine_min_val, migraine_max_val)
    else:
        # Non-migraine days: create substantial overlap
        if np.random.random() < lower_chance:
            # Use lower range (but still high enough to overlap with migraine)
            return np.random.uniform(non_migraine_min_val, non_migraine_lower_max_val)
        else:
            # Use upper range (normal/high mood)
            return np.random.uniform(0.5, non_migraine_upper_max_val)


def generate_timestamps(start_date: datetime, n_days: int, 
                       hour_variation: bool = True) -> pd.Series:
    """
    Generate timestamps for data points.
    
    Args:
        start_date: Starting date
        n_days: Number of days to generate
        hour_variation: Whether to vary the hour of day
    
    Returns:
        Series of timestamps
    """
    timestamps = []
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        
        if hour_variation:
            # Vary hour throughout the day (more realistic)
            hour = np.random.randint(6, 22)  # Between 6 AM and 10 PM
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            timestamp = current_date.replace(hour=hour, minute=minute, second=second)
        else:
            # Fixed time (e.g., end of day)
            timestamp = current_date.replace(hour=18, minute=0, second=0)
        
        timestamps.append(timestamp)
    
    return pd.Series(timestamps)


def calculate_correlation_factor(value: float, target_value: float, 
                                correlation_strength: float = 0.5) -> float:
    """
    Calculate a correlation factor between a value and target value.
    
    Args:
        value: Current value
        target_value: Target value to correlate with
        correlation_strength: Strength of correlation (0-1)
    
    Returns:
        Correlation factor
    """
    # Normalize both values to 0-1 range for comparison
    # This is a simplified correlation model
    normalized_value = (value - 0) / (1 - 0) if value != 0 else 0
    normalized_target = (target_value - 0) / (1 - 0) if target_value != 0 else 0
    
    # Calculate correlation factor
    difference = abs(normalized_value - normalized_target)
    factor = 1.0 - (difference * correlation_strength)
    
    return max(0.0, min(1.0, factor))


def get_hour_from_timestamp(timestamp: datetime) -> int:
    """
    Extract hour from timestamp.
    
    Args:
        timestamp: Datetime object
    
    Returns:
        Hour of day (0-23)
    """
    return timestamp.hour


def get_day_of_week(timestamp: datetime) -> int:
    """
    Get day of week from timestamp.
    
    Args:
        timestamp: Datetime object
    
    Returns:
        Day of week (0=Monday, 6=Sunday)
    """
    return timestamp.weekday()


def create_person_id() -> str:
    """
    Generate a UUID-like person ID.
    
    Returns:
        UUID string
    """
    import uuid
    return str(uuid.uuid4())


def validate_data_range(value: float, min_val: float, max_val: float) -> float:
    """
    Validate and clip value to range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clipped value
    """
    return max(min_val, min(max_val, value))

