"""
Results module for GP ranking experiments.

Provides:
- ModelResult: Dataclass for storing model results
- FailureRecord: Dataclass for tracking failed experiments
- ResultsCollector: Collector for aggregating and saving best models
"""

from .types import ModelResult, FailureRecord
from .collector import ResultsCollector

__all__ = [
    'ModelResult',
    'FailureRecord',
    'ResultsCollector',
]
