"""
PairwiseGP trainer.
Extracted from module_1.py lines 236-401.
"""
import torch
import numpy as np
from typing import Tuple, List, Any, Optional

from botorch.models import PairwiseGP
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

from .base import BaseTrainer
from ..models import build_kernel
from ..solvers import get_optimizer
from ..datatools import get_comparisons


class PairwiseGPTrainer(BaseTrainer):
    """Trainer for PairwiseGP models."""

    @property
    def model_name(self) -> str:
        return "PairwiseGP"

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        comparisons: torch.Tensor = None,
        X_train_pairwise: torch.Tensor = None,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        val_comparisons: Optional[torch.Tensor] = None,
        X_val_pairwise: Optional[torch.Tensor] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Train PairwiseGP model.

        Args:
            X_train: Full training inputs.
            y_train: Training targets (used for comparison generation if needed).
            comparisons: Pre-computed pairwise comparisons.
            X_train_pairwise: Subset of X_train used in the comparisons.
            X_val: Validation inputs (optional).
            y_val: Validation targets (optional).
            val_comparisons: Validation pairwise comparisons (optional).
            X_val_pairwise: Pairwise subset of X_val (optional).

        Returns:
            Tuple of (training_losses, validation_losses).
        """
        # Build model components
        kernel = build_kernel(self.kernel_name, self.dimension)
        self.model = PairwiseGP(
            X_train_pairwise, comparisons,
            covar_module=kernel,
            consolidate_atol=0.0,
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(self.model.likelihood, model=self.model)

        # Move to device
        self.model.to(self.device)
        mll.to(self.device)

        # Setup optimizer
        self.optimizer = get_optimizer(self.optimizer_name, self.model.parameters(), self.lr)

        # Setup validation proxy model (created once, reused each iteration)
        self.val_losses = []
        has_val = X_val_pairwise is not None and val_comparisons is not None
        if has_val:
            val_kernel = build_kernel(self.kernel_name, self.dimension)
            val_proxy = PairwiseGP(
                X_val_pairwise, val_comparisons,
                covar_module=val_kernel,
                consolidate_atol=0.0,
            ).double()
            val_proxy.to(self.device)
            val_mll = PairwiseLaplaceMarginalLogLikelihood(val_proxy.likelihood, val_proxy)

        # Training loop
        self.model.train()
        self.losses = []

        for i in range(self.training_iters):
            if "bfgs" in self.optimizer_name.lower():
                def closure():
                    self.optimizer.zero_grad(set_to_none=True)
                    output = self.model(self.model.datapoints)
                    loss = -mll(output, self.model.train_targets)
                    loss.backward()
                    return loss
                loss = self.optimizer.step(closure)
            else:
                self.optimizer.zero_grad()
                output = self.model(self.model.datapoints)
                loss = -mll(output, self.model.train_targets)
                loss.backward()
                self.optimizer.step()

            self.losses.append(loss.item())

            # Compute validation NLL with current hyperparameters
            if has_val:
                val_proxy.covar_module.load_state_dict(
                    self.model.covar_module.state_dict()
                )
                val_proxy.train()
                with torch.no_grad():
                    val_output = val_proxy(val_proxy.datapoints)
                    val_loss = -val_mll(val_output, val_proxy.train_targets).item()
                self.val_losses.append(val_loss)

            if (i + 1) % 20 == 0:
                try:
                    ls = self.model.covar_module.base_kernel.lengthscale.item()
                    print(f"  Iter {i+1} - Loss: {loss.item():.4f} - Lengthscale: {ls:.3f}")
                except AttributeError:
                    print(f"  Iter {i+1} - Loss: {loss.item():.4f}")

        # Cleanup validation proxy
        if has_val:
            del val_proxy, val_mll, val_kernel

        # Store mll for later use
        self._mll = mll

        return self.losses, self.val_losses

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with trained model."""
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X.to(self.device))
            mean = posterior.mean.squeeze().cpu().numpy()
            variance = posterior.variance.squeeze().detach().cpu().numpy()
        return mean, variance

    def compute_nll(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        comparisons: torch.Tensor = None,
    ) -> float:
        """
        Compute test NLL using a proxy model approach.

        Creates a proxy PairwiseGP with test data structure and copies
        the learned hyperparameters from the trained model.
        """
        # Create test comparisons from true test labels
        test_comparisons = get_comparisons(y_test)

        # Create proxy model with test data structure
        proxy_kernel = build_kernel(self.kernel_name, self.dimension)
        model_proxy = PairwiseGP(
            X_test, test_comparisons,
            covar_module=proxy_kernel,
            consolidate_atol=0.0,
        ).double()

        # Copy learned hyperparameters from trained model
        model_proxy.covar_module.load_state_dict(self.model.covar_module.state_dict())
        mll_proxy = PairwiseLaplaceMarginalLogLikelihood(model_proxy.likelihood, model_proxy)

        # Evaluate NLL in train mode
        model_proxy.train()
        with torch.no_grad():
            output = model_proxy(model_proxy.datapoints)
            nll = -mll_proxy(output, model_proxy.train_targets).item()

        # Cleanup proxy model
        del model_proxy, mll_proxy, proxy_kernel

        return nll

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
        """Release PairwiseGP-specific resources."""
        if hasattr(self, '_mll'):
            del self._mll
        super().cleanup()
