"""
PairwiseGP trainer.

Uses PairwiseGPModel wrapper with outputscale prior.
"""
import torch
import numpy as np
from typing import Tuple, List

from botorch.models import PairwiseGP
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

from .base import BaseTrainer
from ..models.pairwise_gp import PairwiseGPModel
from ..models.kernels import build_kernel
from ..data.dataset import ExperimentData
from ..data.comparisons import get_comparisons
from ..solvers import get_optimizer


class PairwiseGPTrainer(BaseTrainer):
    """
    Trainer for PairwiseGP models.

    Handles training loop with validation MLL tracking using proxy models.
    """

    def __init__(
        self,
        model: PairwiseGPModel,
        training_iters: int,
        lr: float,
        optimizer_name: str,
    ):
        """
        Initialize PairwiseGP trainer.

        Args:
            model: PairwiseGPModel wrapper (must be built before training).
            training_iters: Number of training iterations.
            lr: Learning rate.
            optimizer_name: Name of optimizer.
        """
        super().__init__(model, training_iters, lr, optimizer_name)
        self._mll = None

    def train(self, data: ExperimentData) -> Tuple[List[float], List[float]]:
        """
        Train PairwiseGP model.

        Args:
            data: ExperimentData with prepared train/val splits and comparisons.

        Returns:
            Tuple of (train_losses, val_losses) per iteration.
        """
        device = self.model.device

        # Setup MLL
        self._mll = PairwiseLaplaceMarginalLogLikelihood(
            self.model.model.likelihood, self.model.model
        )
        self._mll.to(device)

        # Setup optimizer
        self.optimizer = get_optimizer(
            self.optimizer_name,
            self.model.model.parameters(),
            self.lr
        )

        # Setup validation proxy model
        val_kernel = build_kernel(self.model.kernel_name, self.model.dimension)
        val_proxy = PairwiseGP(
            data.X_val_pairwise.to(device, dtype=torch.float64),
            data.comparisons_val.to(device),
            covar_module=val_kernel,
            consolidate_atol=0.0,
        ).double()
        val_proxy.to(device)
        val_mll = PairwiseLaplaceMarginalLogLikelihood(val_proxy.likelihood, val_proxy)

        # Training loop
        self.model.model.train()
        self.losses = []
        self.val_losses = []

        for i in range(self.training_iters):
            if "bfgs" in self.optimizer_name.lower():
                def closure():
                    self.optimizer.zero_grad(set_to_none=True)
                    output = self.model.model(self.model.model.datapoints)
                    loss = -self._mll(output, self.model.model.train_targets)
                    loss.backward()
                    return loss
                loss = self.optimizer.step(closure)
            else:
                self.optimizer.zero_grad()
                output = self.model.model(self.model.model.datapoints)
                loss = -self._mll(output, self.model.model.train_targets)
                loss.backward()
                self.optimizer.step()

            self.losses.append(loss.item())

            # Compute validation MLL with current hyperparameters
            # Use strict=False to ignore prior parameters that don't exist in val_proxy
            val_proxy.covar_module.load_state_dict(
                self.model.model.covar_module.state_dict(), strict=False
            )
            val_proxy.train()
            with torch.no_grad():
                val_output = val_proxy(val_proxy.datapoints)
                val_loss = val_mll(val_output, val_proxy.train_targets).item()
            self.val_losses.append(val_loss)

            if (i + 1) % 20 == 0:
                ls = self.model.get_lengthscale()
                print(f"  Iter {i+1} - Loss: {loss.item():.4f} - Lengthscale: {ls:.3f}")

        # Cleanup validation proxy
        del val_proxy, val_mll, val_kernel

        return self.losses, self.val_losses

    def compute_mll(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        comparisons: torch.Tensor = None,
        X_pairwise: torch.Tensor = None,
    ) -> float:
        """
        Compute MLL on given data using proxy model.

        Creates a proxy PairwiseGP with test data structure and copies
        the learned hyperparameters from the trained model.

        Args:
            X: Full input tensor (used if comparisons not provided).
            y: Target tensor (used if comparisons not provided).
            comparisons: Pre-computed comparisons (optional).
            X_pairwise: Pre-computed pairwise X (optional).

        Returns:
            MLL value (higher is better).
        """
        device = self.model.device

        # Generate comparisons if not provided
        if comparisons is None:
            comparisons, X_pairwise = get_comparisons(y, X=X)

        # Create proxy model with test data structure
        proxy_kernel = build_kernel(self.model.kernel_name, self.model.dimension)
        model_proxy = PairwiseGP(
            X_pairwise.to(device, dtype=torch.float64),
            comparisons.to(device),
            covar_module=proxy_kernel,
            consolidate_atol=0.0,
        ).double()
        model_proxy.to(device)

        # Copy learned hyperparameters from trained model
        # Use strict=False to ignore prior parameters
        model_proxy.covar_module.load_state_dict(
            self.model.model.covar_module.state_dict(), strict=False
        )
        mll_proxy = PairwiseLaplaceMarginalLogLikelihood(model_proxy.likelihood, model_proxy)

        # Evaluate MLL in train mode
        model_proxy.train()
        with torch.no_grad():
            output = model_proxy(model_proxy.datapoints)
            mll = mll_proxy(output, model_proxy.train_targets).item()

        # Cleanup proxy model
        del model_proxy, mll_proxy, proxy_kernel

        return mll

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with trained model."""
        return self.model.predict(X)

    def get_lengthscale(self) -> float:
        """Get learned lengthscale."""
        return self.model.get_lengthscale()

    def get_noise_variance(self) -> float:
        """Get noise variance (NaN for PairwiseGP)."""
        return self.model.get_noise_variance()

    def cleanup(self):
        """Release PairwiseGP-specific resources."""
        if self._mll is not None:
            del self._mll
            self._mll = None
        super().cleanup()
