"""
ExactGP trainer.
Extracted from module_1.py lines 403-555.
"""
import torch
import gpytorch
import numpy as np
from typing import Tuple, List, Any

from gpytorch.likelihoods import GaussianLikelihood

from .base import BaseTrainer
from ..models import FlexibleExactGPModel, build_kernel
from ..solvers import get_optimizer


class ExactGPTrainer(BaseTrainer):
    """Trainer for ExactGP models."""

    @property
    def model_name(self) -> str:
        return "ExactGP"

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        comparisons: torch.Tensor = None,
        X_train_pairwise: torch.Tensor = None,
    ) -> List[float]:
        """
        Train ExactGP model.

        Args:
            X_train: Training inputs.
            y_train: Training targets.
            comparisons: Ignored (interface compatibility).
            X_train_pairwise: Ignored (interface compatibility).

        Returns:
            List of training losses.
        """
        # Prepare data on device
        X = X_train.to(self.device, dtype=torch.double)
        y = y_train.to(self.device, dtype=torch.double).squeeze(-1)

        # Build model components
        self.likelihood = GaussianLikelihood().to(self.device, dtype=torch.double)
        kernel = build_kernel(self.kernel_name, self.dimension)
        self.model = FlexibleExactGPModel(X, y, self.likelihood, kernel)
        self.model.to(self.device, dtype=torch.double)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        mll.to(self.device, dtype=torch.double)

        # Setup optimizer
        self.optimizer = get_optimizer(self.optimizer_name, self.model.parameters(), self.lr)

        # Training loop
        self.model.train()
        self.likelihood.train()
        self.losses = []

        # Store training data references for later NLL computation
        self._X_train = X
        self._y_train = y
        self._mll = mll

        for i in range(self.training_iters):
            if "bfgs" in self.optimizer_name.lower():
                def closure():
                    self.optimizer.zero_grad(set_to_none=True)
                    output = self.model(X)
                    loss = -mll(output, y).sum()
                    loss.backward()
                    return loss
                loss = self.optimizer.step(closure)
                self.optimizer.step(closure)  # Double step as in original
            else:
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = -mll(output, y).sum()
                loss.backward()
                self.optimizer.step()

            self.losses.append(loss.item())

            if (i + 1) % 20 == 0:
                try:
                    ls = self.model.covar_module.base_kernel.lengthscale.item()
                    print(f"  Iter {i+1} - Loss: {loss.item():.4f} - Lengthscale: {ls:.3f}")
                except AttributeError:
                    print(f"  Iter {i+1} - Loss: {loss.item():.4f}")

        return self.losses

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with trained model."""
        self.model.eval()
        self.likelihood.eval()

        X_device = X.to(self.device, dtype=torch.double)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(X_device)
            mean = posterior.mean.cpu().numpy()
            variance = posterior.variance.cpu().numpy()

        return mean, variance

    def compute_nll(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        comparisons: torch.Tensor = None,
    ) -> float:
        """Compute NLL on given data."""
        self.model.eval()
        self.likelihood.eval()

        X_device = X.to(self.device, dtype=torch.double)
        y_device = y.to(self.device, dtype=torch.double)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(X_device))
            nll = -pred_dist.log_prob(y_device).sum().item()

        return nll

    def compute_train_nll(self) -> float:
        """Compute NLL on training data (convenience method)."""
        return self.compute_nll(self._X_train, self._y_train)

    def get_lengthscale(self) -> Any:
        """Extract lengthscale from trained model."""
        try:
            ls_tensor = self.model.covar_module.base_kernel.lengthscale
            ls = ls_tensor.detach().cpu().numpy().flatten()
            if ls.size == 1:
                return ls.item()
            return str(ls.tolist())
        except AttributeError:
            return np.nan

    def cleanup(self):
        """Release ExactGP-specific resources."""
        if hasattr(self, 'likelihood'):
            del self.likelihood
        if hasattr(self, '_X_train'):
            del self._X_train
        if hasattr(self, '_y_train'):
            del self._y_train
        if hasattr(self, '_mll'):
            del self._mll
        super().cleanup()
