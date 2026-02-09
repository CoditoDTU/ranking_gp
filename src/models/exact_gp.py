"""
ExactGP model implementation with SNR-based noise prior.
"""
import torch
import gpytorch
import numpy as np
from typing import Tuple, Optional
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

from .base import BaseModelWrapper
from .kernels import build_kernel


class FlexibleExactGPModel(gpytorch.models.ExactGP):
    """
    Flexible ExactGP model that accepts any covariance module.
    """

    def __init__(self, train_x, train_y, likelihood, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class ExactGPModel(BaseModelWrapper):
    """
    ExactGP model wrapper with SNR-based noise prior.

    Uses GammaPrior on likelihood noise variance where:
    - Expected noise: sigma^2_noise = sigma^2_signal / snr_model
    - Prior: Gamma(alpha=2, beta=2/expected_noise_variance)
    """

    @property
    def name(self) -> str:
        return "ExactGP"

    def build(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        comparisons: Optional[torch.Tensor] = None,
        X_pairwise: Optional[torch.Tensor] = None
    ) -> 'ExactGPModel':
        """
        Build ExactGP with GammaPrior on noise.

        Args:
            X_train: Training inputs.
            y_train: Training targets.
            comparisons: Ignored (for interface compatibility).
            X_pairwise: Ignored (for interface compatibility).

        Returns:
            self for method chaining.
        """
        X = X_train.to(self.device, dtype=torch.float64)
        y = y_train.to(self.device, dtype=torch.float64).squeeze(-1)

        # GammaPrior on noise: mean = expected_noise_variance
        # Gamma(alpha, beta) has mean = alpha / beta
        # So beta = alpha / expected_mean
        alpha = 2.0
        beta = alpha / self.expected_noise_variance

        self.likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(concentration=alpha, rate=beta),
            noise_constraint=GreaterThan(1e-6),
        ).to(self.device, dtype=torch.float64)

        kernel = build_kernel(self.kernel_name, self.dimension)
        self.model = FlexibleExactGPModel(X, y, self.likelihood, kernel)
        self.model.to(self.device, dtype=torch.float64)

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
        self.likelihood.eval()
        X_device = X.to(self.device, dtype=torch.float64)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(X_device)
            mean = posterior.mean.cpu().numpy()
            variance = posterior.variance.cpu().numpy()

        return mean, variance

    def get_lengthscale(self) -> float:
        """Get learned lengthscale."""
        try:
            ls = self.model.covar_module.base_kernel.lengthscale
            return ls.detach().cpu().numpy().flatten()[0]
        except AttributeError:
            return np.nan

    def get_noise_variance(self) -> float:
        """Get learned noise variance from likelihood."""
        return self.likelihood.noise.item()
