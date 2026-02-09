"""
Kernel construction utilities for GP models.

Moved from src/models.py - lengthscale is learnable by default.
"""
import torch
from typing import Optional
from gpytorch.kernels import (
    ScaleKernel, MaternKernel, RBFKernel,
    PeriodicKernel, LinearKernel
)
from gpytorch.priors import Prior


def build_kernel(
    kernel_name: str,
    input_dim: int,
    outputscale_prior: Optional[Prior] = None,
) -> ScaleKernel:
    """
    Build a GPyTorch covariance module (kernel).

    Lengthscale is learnable by default.

    Args:
        kernel_name: Name of the kernel type.
        input_dim: Input dimensionality (for reference, not used currently).
        outputscale_prior: Optional prior for the outputscale parameter.

    Returns:
        ScaleKernel wrapping the base kernel.

    Raises:
        ValueError: If kernel name is unknown.
    """
    if kernel_name == 'squared_exponential':
        base_kernel = RBFKernel()
    elif kernel_name == 'matern_5_2':
        base_kernel = MaternKernel(nu=2.5)
    elif kernel_name == 'matern_3_2':
        base_kernel = MaternKernel(nu=1.5)
    elif kernel_name == 'exponential':
        base_kernel = MaternKernel(nu=0.5)
    elif kernel_name == 'periodic':
        base_kernel = PeriodicKernel()
    elif kernel_name == 'linear':
        base_kernel = LinearKernel()
    else:
        raise ValueError(f"Unknown kernel name: '{kernel_name}'.")

    # Wrap in ScaleKernel with optional outputscale prior
    if outputscale_prior is not None:
        return ScaleKernel(base_kernel, outputscale_prior=outputscale_prior)
    else:
        return ScaleKernel(base_kernel)