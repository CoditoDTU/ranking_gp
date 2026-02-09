"""
ExactGP trainer.

Uses ExactGPModel wrapper with GammaPrior on noise.
"""
import torch
import gpytorch
import numpy as np
from typing import Tuple, List

from .base import BaseTrainer
from ..models.exact_gp import ExactGPModel
from ..data.dataset import ExperimentData
from ..solvers import get_optimizer


class ExactGPTrainer(BaseTrainer):
    """
    Trainer for ExactGP models.

    Handles training loop with validation MLL tracking.
    """

    def __init__(
        self,
        model: ExactGPModel,
        training_iters: int,
        lr: float,
        optimizer_name: str,
    ):
        """
        Initialize ExactGP trainer.

        Args:
            model: ExactGPModel wrapper (must be built before training).
            training_iters: Number of training iterations.
            lr: Learning rate.
            optimizer_name: Name of optimizer.
        """
        super().__init__(model, training_iters, lr, optimizer_name)
        self._mll = None

    def train(self, data: ExperimentData) -> Tuple[List[float], List[float]]:
        """
        Train ExactGP model.

        Args:
            data: ExperimentData with prepared train/val splits.

        Returns:
            Tuple of (train_losses, val_losses) per iteration.
        """
        device = self.model.device

        # Get data tensors
        X_train = data.X_train.to(device, dtype=torch.float64)
        y_train = data.Y_train_noisy.to(device, dtype=torch.float64).squeeze(-1)
        X_val = data.X_val.to(device, dtype=torch.float64)
        y_val = data.Y_val_noisy.to(device, dtype=torch.float64).squeeze(-1)

        # Setup MLL
        self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model.model
        )
        self._mll.to(device, dtype=torch.float64)

        # Setup optimizer
        self.optimizer = get_optimizer(
            self.optimizer_name,
            self.model.model.parameters(),
            self.lr
        )

        # Training loop
        self.model.model.train()
        self.model.likelihood.train()
        self.losses = []
        self.val_losses = []

        for i in range(self.training_iters):
            if "bfgs" in self.optimizer_name.lower():
                def closure():
                    self.optimizer.zero_grad(set_to_none=True)
                    output = self.model.model(X_train)
                    loss = -self._mll(output, y_train).sum()
                    loss.backward()
                    return loss
                loss = self.optimizer.step(closure)
                self.optimizer.step(closure)  # Double step for BFGS
            else:
                self.optimizer.zero_grad()
                output = self.model.model(X_train)
                loss = -self._mll(output, y_train).sum()
                loss.backward()
                self.optimizer.step()

            self.losses.append(loss.item())

            # Compute validation MLL
            self.model.model.eval()
            self.model.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                val_pred_dist = self.model.likelihood(self.model.model(X_val))
                val_mll = val_pred_dist.log_prob(y_val).sum().item()
            self.val_losses.append(val_mll)
            self.model.model.train()
            self.model.likelihood.train()

            if (i + 1) % 20 == 0:
                ls = self.model.get_lengthscale()
                print(f"  Iter {i+1} - Loss: {loss.item():.4f} - Lengthscale: {ls:.3f}")

        return self.losses, self.val_losses

    def compute_mll(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        comparisons: torch.Tensor = None,
        X_pairwise: torch.Tensor = None,
    ) -> float:
        """
        Compute MLL on given data.

        Args:
            X: Input tensor.
            y: Target tensor.
            comparisons: Ignored.
            X_pairwise: Ignored.

        Returns:
            MLL value (higher is better).
        """
        device = self.model.device
        self.model.model.eval()
        self.model.likelihood.eval()

        X_device = X.to(device, dtype=torch.float64)
        y_device = y.to(device, dtype=torch.float64).squeeze(-1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.model.likelihood(self.model.model(X_device))
            mll = pred_dist.log_prob(y_device).sum().item()

        return mll

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with trained model."""
        return self.model.predict(X)

    def get_lengthscale(self) -> float:
        """Get learned lengthscale."""
        return self.model.get_lengthscale()

    def get_noise_variance(self) -> float:
        """Get learned noise variance."""
        return self.model.get_noise_variance()

    def cleanup(self):
        """Release ExactGP-specific resources."""
        if self._mll is not None:
            del self._mll
            self._mll = None
        super().cleanup()
