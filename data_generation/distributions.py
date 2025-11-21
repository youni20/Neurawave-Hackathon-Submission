"""
Distribution configuration and utilities for synthetic data generation.
Provides statistical distributions for step count, mood, screen brightness, and trigger severities.
"""

import numpy as np
from scipy import stats
from scipy.stats import truncnorm


class DistributionConfig:
    """Configuration for statistical distributions used in data generation."""
    
    # Step Count Distribution (Truncated Normal)
    STEP_COUNT_MEAN = 8000
    STEP_COUNT_STD = 3000
    STEP_COUNT_MIN = 0
    STEP_COUNT_MAX = 25000
    
    # Mood Distribution (Beta, scaled to 0-10)
    MOOD_ALPHA = 2.0
    MOOD_BETA = 2.0
    MOOD_MIN = 0.0
    MOOD_MAX = 10.0
    
    # Screen Brightness Distribution (Gamma)
    BRIGHTNESS_SHAPE = 2.0
    BRIGHTNESS_SCALE = 50.0
    BRIGHTNESS_MIN = 0.0
    BRIGHTNESS_MAX = 500.0
    
    # Trigger Severity Distribution (Beta per category)
    TRIGGER_ALPHA = 1.5
    TRIGGER_BETA = 2.5


def generate_step_count(n_samples=1, weekday_factor=1.0, migraine_factor=1.0, 
                        random_state=None, noise_scale=0.15, params=None):
    """
    Generate step count using truncated normal distribution.
    
    Args:
        n_samples: Number of samples to generate
        weekday_factor: Multiplier for weekday/weekend (1.0 = weekday, 0.7-0.8 = weekend)
        migraine_factor: Multiplier for migraine days (0.6-0.8 = lower activity on migraine days)
        random_state: Random seed for reproducibility
    
    Returns:
        Array of step counts
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate bounds for truncated normal
    a = (DistributionConfig.STEP_COUNT_MIN - DistributionConfig.STEP_COUNT_MEAN) / DistributionConfig.STEP_COUNT_STD
    b = (DistributionConfig.STEP_COUNT_MAX - DistributionConfig.STEP_COUNT_MEAN) / DistributionConfig.STEP_COUNT_STD
    
    # Generate truncated normal samples
    samples = truncnorm.rvs(a, b, 
                            loc=DistributionConfig.STEP_COUNT_MEAN,
                            scale=DistributionConfig.STEP_COUNT_STD,
                            size=n_samples,
                            random_state=random_state)
    
    # Apply factors
    samples = samples * weekday_factor * migraine_factor
    
    # PHASE 0: Add random noise to step_count (symptom feature) to break separability
    # Add substantial noise to create overlap between migraine and non-migraine days
    # Use params if provided, otherwise use noise_scale parameter
    if params is not None:
        noise_scale_val = params.step_count_noise_scale
    else:
        noise_scale_val = noise_scale
    noise = np.random.normal(0, DistributionConfig.STEP_COUNT_MEAN * noise_scale_val, size=n_samples)
    samples = samples + noise
    
    # Ensure within bounds
    samples = np.clip(samples, DistributionConfig.STEP_COUNT_MIN, DistributionConfig.STEP_COUNT_MAX)
    
    return np.round(samples).astype(int)


def generate_mood(n_samples=1, day_of_week=0, migraine_factor=1.0, random_state=None,
                 noise_scale=7.5, uniform_noise_range=2.5, params=None):
    """
    Generate mood scores using beta distribution scaled to 0-10.
    
    SOLUTION 3: Symptom features should be mostly independent of migraine status
    with only a small bias. Heavy noise ensures substantial overlap.
    
    Args:
        n_samples: Number of samples to generate
        day_of_week: Day of week (0=Monday, 6=Sunday) - affects mood
        migraine_factor: Multiplier for migraine days (should be close to 1.0 for overlap)
        random_state: Random seed for reproducibility
    
    Returns:
        Array of mood scores (0-10)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate base mood from beta distribution (independent of migraine)
    beta_samples = np.random.beta(DistributionConfig.MOOD_ALPHA, 
                                  DistributionConfig.MOOD_BETA, 
                                  size=n_samples)
    
    # Scale to 0-10
    mood_scores = beta_samples * DistributionConfig.MOOD_MAX
    
    # Apply day of week effect (lower on Mondays)
    monday_penalty = 0.85 if day_of_week == 0 else 1.0
    mood_scores = mood_scores * monday_penalty
    
    # Apply migraine factor (small effect to create slight bias, but with overlap)
    # The factor ranges are designed to overlap, so this creates minimal separation
    mood_scores = mood_scores * migraine_factor
    
    # Add VERY significant random noise to create substantial overlap
    # PHASE 0: Drastically increase noise to break symptom feature separability
    # Noise should be large enough to completely overwhelm the migraine bias
    # Use params if provided, otherwise use noise_scale/uniform_noise_range parameters
    if params is not None:
        noise_scale_val = params.mood_noise_scale
        uniform_noise_range_val = params.mood_uniform_noise_range
    else:
        noise_scale_val = noise_scale
        uniform_noise_range_val = uniform_noise_range
    
    noise = np.random.normal(0, noise_scale_val, size=n_samples)
    mood_scores = mood_scores + noise
    
    # Add additional uniform noise to ensure maximum overlap
    uniform_noise = np.random.uniform(-uniform_noise_range_val, uniform_noise_range_val, size=n_samples)
    mood_scores = mood_scores + uniform_noise
    
    # Ensure within bounds
    mood_scores = np.clip(mood_scores, DistributionConfig.MOOD_MIN, DistributionConfig.MOOD_MAX)
    
    return mood_scores


def generate_screen_brightness(n_samples=1, hour_of_day=12, random_state=None):
    """
    Generate screen brightness using gamma distribution.
    
    Args:
        n_samples: Number of samples to generate
        hour_of_day: Hour of day (0-23) - affects brightness (higher during day)
        random_state: Random seed for reproducibility
    
    Returns:
        Array of brightness values (0-500 nits)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate gamma distribution samples
    gamma_samples = np.random.gamma(DistributionConfig.BRIGHTNESS_SHAPE,
                                    DistributionConfig.BRIGHTNESS_SCALE,
                                    size=n_samples)
    
    # Apply temporal pattern (higher during day, lower at night)
    # Peak around 12-16 (noon to afternoon), lower at night (22-6)
    if 6 <= hour_of_day <= 22:
        # Daytime: higher brightness
        time_factor = 0.7 + 0.3 * np.sin((hour_of_day - 6) * np.pi / 16)
    else:
        # Nighttime: much lower brightness
        time_factor = 0.2 + 0.1 * np.sin((hour_of_day + 2) * np.pi / 8)
    
    brightness = gamma_samples * time_factor
    
    # Ensure within bounds
    brightness = np.clip(brightness, DistributionConfig.BRIGHTNESS_MIN, DistributionConfig.BRIGHTNESS_MAX)
    
    return brightness


def generate_trigger_severity(n_samples=1, category='stress', random_state=None):
    """
    Generate trigger severity scores using beta distribution.
    
    Args:
        n_samples: Number of samples to generate
        category: Trigger category (affects distribution parameters)
        random_state: Random seed for reproducibility
    
    Returns:
        Array of trigger severity scores (0-1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Different categories may have different distributions
    # For now, use base beta distribution
    alpha = DistributionConfig.TRIGGER_ALPHA
    beta = DistributionConfig.TRIGGER_BETA
    
    # Generate beta distribution samples (0-1)
    severity_scores = np.random.beta(alpha, beta, size=n_samples)
    
    # Ensure within bounds
    severity_scores = np.clip(severity_scores, 0.0, 1.0)
    
    return severity_scores


def normalize_step_count(step_counts, method='log'):
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


def get_mood_category(mood_score):
    """
    Categorize mood score into descriptive categories.
    
    Args:
        mood_score: Mood score (0-10)
    
    Returns:
        Mood category string
    """
    if mood_score <= 2:
        return "Very Low"
    elif mood_score <= 4:
        return "Low"
    elif mood_score <= 6:
        return "Moderate"
    elif mood_score <= 8:
        return "Good"
    else:
        return "Very Good"

