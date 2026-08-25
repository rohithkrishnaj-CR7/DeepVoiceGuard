"""
Model training and academic evaluation metric utilities.
"""
from .metrics import calculate_metrics, compute_eer
from .trainer import ModelTrainer

__all__ = [
    'calculate_metrics',
    'compute_eer',
    'ModelTrainer'
]
