"""
Trainer module for GP ranking experiments.

Provides:
- BaseTrainer: Abstract base class for trainers
- ExactGPTrainer: Trainer for ExactGP with noise prior
- PairwiseGPTrainer: Trainer for PairwiseGP with outputscale prior
"""
from .base import BaseTrainer
from .exact_gp import ExactGPTrainer
from .pairwise_gp import PairwiseGPTrainer

__all__ = [
    'BaseTrainer',
    'ExactGPTrainer',
    'PairwiseGPTrainer',
]
