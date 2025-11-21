"""
Migraine probability calculation based on predictor features.
This module calculates the probability of migraine occurrence based on
predictor features (stress, triggers, weather, etc.) rather than symptoms.
"""

import numpy as np
from typing import Dict, Optional


def calculate_migraine_probability(
    stress_intensity: float,
    trigger_stress: float,
    trigger_hormonal: float,
    trigger_sleep: float,
    trigger_weather: float,
    trigger_food: float,
    trigger_sensory: float,
    trigger_physical: float,
    weather_pressure: float,
    weather_temp: float,
    day_of_week: int,
    consecutive_migraine_days: int = 0,  # PHASE 1: Temporal momentum
    days_since_last_migraine: int = 0,  # PHASE 1: Days since last migraine
    base_probability: float = 0.08,  # Base daily probability (~2.4 migraines/month)
    random_state: Optional[int] = None
) -> float:
    """
    Calculate migraine probability based on predictor features.
    
    This function models realistic migraine triggers:
    - High stress increases probability
    - Trigger levels contribute to probability
    - Weather changes (pressure, temperature) can trigger
    - Day of week effects (some people have patterns)
    - PHASE 1: Temporal momentum (consecutive migraine days)
    
    Args:
        stress_intensity: Stress level (0-10)
        trigger_stress: Stress trigger sensitivity (0-1)
        trigger_hormonal: Hormonal trigger sensitivity (0-1)
        trigger_sleep: Sleep trigger sensitivity (0-1)
        trigger_weather: Weather trigger sensitivity (0-1)
        trigger_food: Food trigger sensitivity (0-1)
        trigger_sensory: Sensory trigger sensitivity (0-1)
        trigger_physical: Physical trigger sensitivity (0-1)
        weather_pressure: Atmospheric pressure (hPa)
        weather_temp: Temperature (Celsius)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        consecutive_migraine_days: Number of consecutive days with migraine (PHASE 1)
        days_since_last_migraine: Days since last migraine episode (PHASE 1)
        base_probability: Base daily migraine probability
        random_state: Random seed
    
    Returns:
        Probability of migraine (0-1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Start with base probability
    prob = base_probability
    
    # PHASE 2: Strengthen logical predictor contributions
    # Stress contribution (normalized 0-10 to 0-1)
    stress_factor = stress_intensity / 10.0
    prob += stress_factor * trigger_stress * 0.25  # PHASE 2: Increased from 0.15 to 0.25
    
    # PHASE 2: Strengthen trigger contributions
    trigger_contribution = (
        trigger_stress * 0.12 +
        trigger_hormonal * 0.10 +
        trigger_sleep * 0.15 +
        trigger_weather * 0.08 +
        trigger_food * 0.07 +
        trigger_sensory * 0.09 +
        trigger_physical * 0.06
    )
    prob += trigger_contribution * 0.30  # PHASE 2: Increased from 0.20 to 0.30
    
    # Weather effects
    # Pressure changes (normalized around 1012 hPa)
    pressure_deviation = abs(weather_pressure - 1012.0) / 20.0  # Normalize
    prob += pressure_deviation * trigger_weather * 0.05
    
    # Temperature effects (extreme temperatures)
    temp_factor = 0.0
    if weather_temp < 0 or weather_temp > 25:
        temp_factor = min(abs(weather_temp - 12.5) / 25.0, 1.0)
    prob += temp_factor * trigger_weather * 0.03
    
    # PHASE 2: Add sleep quality interaction (if sleep trigger is high)
    if trigger_sleep > 0.5:
        sleep_boost = trigger_sleep * 0.05  # Additional boost for high sleep sensitivity
        prob += sleep_boost
    
    # PHASE 2: Enhanced weather interaction
    weather_interaction = trigger_weather * pressure_deviation * (temp_factor + 0.1)
    prob += weather_interaction * 0.04  # Additional weather effect
    
    # Day of week effects (some people have patterns)
    day_factor = np.random.uniform(-0.02, 0.02)
    prob += day_factor
    
    # PHASE 1: Temporal momentum - consecutive migraine days increase probability
    # Only apply momentum boost if already had 2+ consecutive days (use previous day's value)
    if consecutive_migraine_days >= 2:
        # Reduced momentum: +5% per day (was +10%), capped at +20% (was +30%)
        # This prevents momentum from causing too many migraines
        momentum_boost = min(0.05 * (consecutive_migraine_days - 1), 0.20)
        prob += momentum_boost
    
    # PHASE 1: Days since last migraine - slight reduction if recently had migraine
    if days_since_last_migraine > 0 and days_since_last_migraine <= 3:
        # Recent migraine might reduce probability slightly (recovery period)
        recovery_factor = 0.02 * (4 - days_since_last_migraine)  # Small reduction
        prob -= recovery_factor
    
    # Add some randomness to make it non-deterministic
    noise = np.random.normal(0, 0.03)
    prob += noise
    
    # Ensure probability is in valid range [0, 1]
    prob = np.clip(prob, 0.0, 1.0)
    
    return prob


def sample_migraine_status(probability: float, random_state: Optional[int] = None) -> bool:
    """
    Sample migraine status from probability.
    
    Args:
        probability: Probability of migraine (0-1)
        random_state: Random seed
    
    Returns:
        True if migraine occurs, False otherwise
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    return np.random.random() < probability

