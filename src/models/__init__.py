"""
Models module for GP ranking experiments.

Provides:
- BaseModelWrapper: Abstract base class for GP models
- ExactGPModel: ExactGP with SNR-based noise prior
- PairwiseGPModel: PairwiseGP with outputscale prior
- build_kernel: Kernel construction utility
"""

from .base import BaseModelWrapper
from .exact_gp import ExactGPModel, FlexibleExactGPModel
from .pairwise_gp import PairwiseGPModel
from .kernels import build_kernel

__all__ = [
    'BaseModelWrapper',
    'ExactGPModel',
    'FlexibleExactGPModel',
    'PairwiseGPModel',
    'build_kernel',
]