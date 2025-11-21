"""
Data generation package for synthetic migraine prediction data.
"""

from .synthetic_data_generator import SyntheticDataGenerator
from .questionnaire_processor import QuestionnaireProcessor
from .ml_trigger_classifier import MLTriggerClassifier
from .distributions import (
    generate_step_count,
    generate_mood,
    generate_screen_brightness,
    normalize_step_count,
    get_mood_category
)

__all__ = [
    'SyntheticDataGenerator',
    'QuestionnaireProcessor',
    'MLTriggerClassifier',
    'generate_step_count',
    'generate_mood',
    'generate_screen_brightness',
    'normalize_step_count',
    'get_mood_category'
]

