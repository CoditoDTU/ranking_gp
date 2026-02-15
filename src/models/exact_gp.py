"""
ExactGP model implementation with SNR-based noise prior.
"""
import torch
import gpytorch
import numpy as np
from typing import Tuple, Optional
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
        self.mean_module = gpytorch.means.ZeroMean()#ConstantMean()
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

        # Estimate initial noise from data variance as a heuristic
        # This helps when snr_model doesn't match snr_data
        data_variance = y.var().item()
        initial_noise = max(data_variance * 0.5, self.expected_noise_variance)

        # No prior on noise - let the data speak
        # The MLL objective naturally regularizes noise estimation
        self.likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-6),
        ).to(self.device, dtype=torch.float64)

        # Initialize noise to a reasonable starting point
        self.likelihood.noise = initial_noise

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

        Returns PREDICTIVE variance (includes observation noise), not just
        posterior variance. This is critical for calibrated uncertainty:
        - Posterior variance Var[f(x)] → uncertainty about the latent function
        - Predictive variance Var[y] = Var[f(x)] + σ²_noise → uncertainty about observations

        Args:
            X: Input tensor.

        Returns:
            Tuple of (mean, predictive_variance) as numpy arrays.
        """
        self.model.eval()
        self.likelihood.eval()
        X_device = X.to(self.device, dtype=torch.float64)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(X_device)
            #mean = posterior.mean.cpu().numpy() # Original
            #variance = posterior.variance.cpu().numpy()
            # Pass through likelihood to get predictive distribution p(y*|x*, D)
            # This adds observation noise: Var[y*] = Var[f(x*)] + σ²_noise
            
            predictive = self.likelihood(posterior) # predictive
            mean = predictive.mean.cpu().numpy() 
            variance = predictive.variance.cpu().numpy()

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
