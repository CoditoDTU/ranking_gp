"""
PairwiseGP trainer.

Uses PairwiseGPModel wrapper with outputscale prior.
"""
import copy
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

    Handles training loop with validation MLL tracking and early stopping.
    """

    def __init__(
        self,
        model: PairwiseGPModel,
        training_iters: int,
        lr: float,
        optimizer_name: str,
        early_stopping: bool = True,
        patience: int = 50,
        min_relative_delta: float = 0.001,
        check_interval: int = 10,
    ):
        """
        Initialize PairwiseGP trainer.

        Args:
            model: PairwiseGPModel wrapper (must be built before training).
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
        Train PairwiseGP model with optional early stopping.

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
        val_mll_fn = PairwiseLaplaceMarginalLogLikelihood(val_proxy.likelihood, val_proxy)

        # Training loop
        self.model.model.train()
        self.losses = []
        self.val_losses = []

        # Early stopping state
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        self.stopped_early = False
        self.best_iter = 0

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

            # Compute validation loss with current hyperparameters
            val_proxy.covar_module.load_state_dict(
                self.model.model.covar_module.state_dict(), strict=False
            )
            val_proxy.train()
            with torch.no_grad():
                val_output = val_proxy(val_proxy.datapoints)
                # Negate to match train_loss convention: lower is better
                val_loss = -val_mll_fn(val_output, val_proxy.train_targets).item()
            self.val_losses.append(val_loss)

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
                    best_model_state = copy.deepcopy(self.model.model.covar_module.state_dict())
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
            self.model.model.covar_module.load_state_dict(best_model_state)
            if not self.stopped_early:
                print(f"  Training complete. Restored best model from iter {self.best_iter}")

        # Cleanup validation proxy
        del val_proxy, val_mll_fn, val_kernel

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
