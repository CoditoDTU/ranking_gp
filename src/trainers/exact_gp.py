"""
ExactGP trainer.

Uses ExactGPModel wrapper with GammaPrior on noise.
"""
import copy
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

    Handles training loop with validation MLL tracking and early stopping.
    """

    def __init__(
        self,
        model: ExactGPModel,
        training_iters: int,
        lr: float,
        optimizer_name: str,
        early_stopping: bool = True,
        patience: int = 50,
        min_relative_delta: float = 0.001,
        check_interval: int = 10,
    ):
        """
        Initialize ExactGP trainer.

        Args:
            model: ExactGPModel wrapper (must be built before training).
            training_iters: Number of training iterations.
            lr: Learning rate.
            optimizer_name: Name of optimizer.
            early_stopping: Whether to use early stopping.
            patience: Iterations without improvement before stopping.
            min_relative_delta: Minimum relative improvement threshold.
            check_interval: Check validation every N iterations.
        """
        super().__init__(model, training_iters, lr, optimizer_name)
        self._mll = None
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_relative_delta = min_relative_delta
        self.check_interval = check_interval
        self.stopped_early = False
        self.best_iter = 0

    def train(self, data: ExperimentData) -> Tuple[List[float], List[float]]:
        """
        Train ExactGP model with optional early stopping.

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

        # Early stopping state
        best_val_loss = float('inf')
        best_model_state = None
        best_likelihood_state = None
        patience_counter = 0
        self.stopped_early = False
        self.best_iter = 0

        for i in range(self.training_iters):
            if "bfgs" in self.optimizer_name.lower():
                def closure():
                    self.optimizer.zero_grad(set_to_none=True)
                    output = self.model.model(X_train)
                    loss = -self._mll(output, y_train)
                    loss.backward()
                    return loss
                loss = self.optimizer.step(closure)
                self.optimizer.step(closure)  # Double step for BFGS
            else:
                self.optimizer.zero_grad()
                output = self.model.model(X_train)
                loss = -self._mll(output, y_train)
                loss.backward()
                self.optimizer.step()

            self.losses.append(loss.item())

            # Compute validation loss
            self.model.model.eval()
            self.model.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                val_pred_dist = self.model.likelihood(self.model.model(X_val))
                # Negate to match train_loss convention: lower is better
                val_loss = -val_pred_dist.log_prob(y_val).sum().item()
            self.val_losses.append(val_loss)
            self.model.model.train()
            self.model.likelihood.train()

            # Early stopping check
            if self.early_stopping and (i + 1) % self.check_interval == 0:
                # Relative improvement: positive means we improved (val_loss decreased)
                if best_val_loss != float('inf'):
                    relative_improvement = (best_val_loss - val_loss) / abs(best_val_loss)
                else:
                    relative_improvement = float('inf')  # First check always counts

                if relative_improvement > self.min_relative_delta:
                    # Improved - save checkpoint
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.model.state_dict())
                    best_likelihood_state = copy.deepcopy(self.model.likelihood.state_dict())
                    patience_counter = 0
                    self.best_iter = i + 1
                else:
                    patience_counter += self.check_interval
                    if patience_counter >= self.patience:
                        print(f"  Early stopping at iter {i+1} (best was iter {self.best_iter})")
                        self.stopped_early = True
                        break

            if (i + 1) % 20 == 0:
                ls = self.model.get_lengthscale()
                print(f"  Iter {i+1} - Loss: {loss.item():.4f} - Val: {val_loss:.4f} - LS: {ls:.3f}")

        # Restore best model if early stopping was used and we have a checkpoint
        if self.early_stopping and best_model_state is not None:
            self.model.model.load_state_dict(best_model_state)
            self.model.likelihood.load_state_dict(best_likelihood_state)
            if not self.stopped_early:
                print(f"  Training complete. Restored best model from iter {self.best_iter}")

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
