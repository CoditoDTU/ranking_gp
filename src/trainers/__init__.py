"""Trainer classes for GP models."""
from .base import BaseTrainer
from .pairwise_gp import PairwiseGPTrainer
from .exact_gp import ExactGPTrainer

__all__ = ['BaseTrainer', 'PairwiseGPTrainer', 'ExactGPTrainer']
