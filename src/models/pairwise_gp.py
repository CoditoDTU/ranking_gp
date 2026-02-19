"""
PairwiseGP model implementation with SNR-based outputscale prior.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from gpytorch.priors import GammaPrior
from botorch.models import PairwiseGP

from .base import BaseModelWrapper
from .kernels import build_kernel


class PairwiseGPModel(BaseModelWrapper):
    """
    PairwiseGP model wrapper with SNR-based outputscale prior.

    Uses GammaPrior on kernel outputscale where:
    - Expected outputscale: sigma^2_signal (signal variance)
    - Prior: Gamma(alpha=2, beta=2/signal_variance)

    Note: PairwiseGP doesn't have explicit noise variance - noise is
    implicit in the pairwise comparisons.
    """

    @property
    def name(self) -> str:
        return "PairwiseGP"

    def build(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        comparisons: Optional[torch.Tensor] = None,
        X_pairwise: Optional[torch.Tensor] = None
    ) -> 'PairwiseGPModel':
        """
        Build PairwiseGP with outputscale prior.

        Args:
            X_train: Ignored (uses X_pairwise instead).
            y_train: Ignored (uses comparisons instead).
            comparisons: Pairwise comparison indices (required).
            X_pairwise: Pairwise input data (required).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If comparisons or X_pairwise is None.
        """
        if comparisons is None or X_pairwise is None:
            raise ValueError("PairwiseGP requires comparisons and X_pairwise")

        # Outputscale prior centered on signal variance
        # Gamma(alpha, beta) has mean = alpha / beta

        kernel = build_kernel(
            self.kernel_name,
            self.dimension
            #outputscale_prior=outputscale_prior
        )

        self.model = PairwiseGP(
            X_pairwise.to(self.device, dtype=torch.float64),
            comparisons.to(self.device),
            covar_module=kernel,
            consolidate_atol=0.0,
        )
        self.model.to(self.device)

        return self

    def forward(self, X: torch.Tensor):
        """Forward pass through the model."""
        return self.model(X.to(self.device, dtype=torch.float64))

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with no gradients in eval mode.

        Args:
            X: Input tensor.

        Returns:
            Tuple of (mean, variance) as numpy arrays.
        """
        self.model.eval()
        X_device = X.to(self.device, dtype=torch.float64)

        with torch.no_grad():
            posterior = self.model.posterior(X_device)
            mean = posterior.mean.squeeze().cpu().numpy()
            variance = posterior.variance.squeeze().detach().cpu().numpy()

        return mean, variance

    def get_lengthscale(self) -> float:
        """Get learned lengthscale."""
        try:
            ls = self.model.covar_module.base_kernel.lengthscale
            return ls.detach().cpu().numpy().flatten()[0]
        except AttributeError:
            return np.nan

    def get_noise_variance(self) -> float:
        """
        Get noise variance.

        PairwiseGP doesn't have explicit noise variance.
        """
        return np.nan
