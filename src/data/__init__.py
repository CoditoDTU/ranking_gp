"""
Data module for GP ranking experiments.

Provides:
- ExperimentData: Main class for data generation, splitting, and noise application
- get_comparisons: Generate pairwise comparisons from y values
- fitness_function: Factory function for benchmark fitness functions
"""

from .dataset import ExperimentData
from .comparisons import get_comparisons
from .fitness_functions import fitness_function

__all__ = [
    'ExperimentData',
    'get_comparisons',
    'fitness_function',
]
