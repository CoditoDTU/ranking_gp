"""
ExperimentData class for handling all data operations.

Handles data generation, splitting, noise application, and comparison generation.
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .fitness_functions import fitness_function
from .comparisons import get_comparisons


@dataclass
class ExperimentData:
    """
    Handles all data generation, splitting, and noise application.

    This class encapsulates all data-related operations that were previously
    scattered in run_experiments.py.

    Usage:
        data = ExperimentData(
            fitness_fn_name='ackley',
            dimension=1,
            n_train=50,
            n_test=100,
            val_fraction=0.2,
            snr=10.0,
            seed=42,
        )
        data.prepare()  # Runs sample(), noise(), compute_comparisons()

    Attributes (after prepare()):
        X_train, X_val, X_test: Input tensors
        Y_train_true, Y_val_true, Y_test_true: Ground truth outputs
        Y_train_noisy, Y_val_noisy, Y_test_noisy: Noisy outputs
        signal_variance: Variance of the signal (computed from Y_train_true)
        noise_variance: Computed noise variance (sigma^2_signal / SNR)
        comparisons_train, X_train_pairwise: Pairwise data for training
        comparisons_val, X_val_pairwise: Pairwise data for validation
    """

    # Constructor arguments
    fitness_fn_name: str
    dimension: int
    n_train: int
    n_test: int
    val_fraction: float
    snr: float
    seed: int

    # Computed after sample()
    fitness_fn: Any = field(init=False, default=None)
    X_train: torch.Tensor = field(init=False, default=None)
    X_val: torch.Tensor = field(init=False, default=None)
    X_test: torch.Tensor = field(init=False, default=None)
    Y_train_true: torch.Tensor = field(init=False, default=None)
    Y_val_true: torch.Tensor = field(init=False, default=None)
    Y_test_true: torch.Tensor = field(init=False, default=None)

    # NumPy versions for results storage
    X_train_np: np.ndarray = field(init=False, default=None)
    X_val_np: np.ndarray = field(init=False, default=None)
    X_test_np: np.ndarray = field(init=False, default=None)

    # Computed after noise()
    Y_train_noisy: torch.Tensor = field(init=False, default=None)
    Y_val_noisy: torch.Tensor = field(init=False, default=None)
    Y_test_noisy: torch.Tensor = field(init=False, default=None)
    signal_variance: float = field(init=False, default=None)
    noise_variance: float = field(init=False, default=None)

    # Computed after compute_comparisons()
    comparisons_train: torch.Tensor = field(init=False, default=None)
    X_train_pairwise: torch.Tensor = field(init=False, default=None)
    comparisons_val: torch.Tensor = field(init=False, default=None)
    X_val_pairwise: torch.Tensor = field(init=False, default=None)

    def sample(self) -> 'ExperimentData':
        """
        Sample from fitness function and split into train/val/test.

        Sets all random seeds and creates the data splits.

        Returns:
            self for method chaining.
        """
        # Set all seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create fitness function
        self.fitness_fn = fitness_function(self.fitness_fn_name, self.dimension)

        # Sample all points
        n_total = self.n_train + self.n_test
        X_all = self.fitness_fn.sample_uniform(n_total, seed=self.seed)
        Y_all_true = torch.tensor(self.fitness_fn.output(X_all), dtype=torch.float64)

        # Split into train and test
        X_train_full = X_all[:self.n_train]
        X_test_np = X_all[self.n_train:]
        self.X_test = torch.tensor(X_test_np, dtype=torch.float64)
        self.X_test_np = X_test_np

        Y_train_full = Y_all_true[:self.n_train]
        self.Y_test_true = Y_all_true[self.n_train:]

        # Split train into train and validation
        n_val = int(self.n_train * self.val_fraction)
        perm = np.random.permutation(self.n_train)
        train_idx = perm[n_val:]
        val_idx = perm[:n_val]

        # Store numpy versions
        self.X_train_np = X_train_full[train_idx]
        self.X_val_np = X_train_full[val_idx]

        # Convert to tensors
        self.X_train = torch.tensor(self.X_train_np, dtype=torch.float64)
        self.X_val = torch.tensor(self.X_val_np, dtype=torch.float64)
        self.Y_train_true = Y_train_full[train_idx]
        self.Y_val_true = Y_train_full[val_idx]

        return self

    def noise(self) -> 'ExperimentData':
        """
        Apply SNR-based noise to all splits.

        Computes noise variance as: sigma^2_noise = sigma^2_signal / SNR
        Then applies Gaussian noise: y_noisy = y_true + N(0, sqrt(sigma^2_noise))

        If SNR is infinity, no noise is applied (y_noisy = y_true).

        Returns:
            self for method chaining.
        """
        # Compute signal variance from training data
        self.signal_variance = self.Y_train_true.var().item()

        if self.snr == float('inf'):
            # No noise case
            self.noise_variance = 0.0
            self.Y_train_noisy = self.Y_train_true.clone()
            self.Y_val_noisy = self.Y_val_true.clone()
            self.Y_test_noisy = self.Y_test_true.clone()
        else:
            # Compute noise variance: sigma^2_noise = sigma^2_signal / SNR
            self.noise_variance = self.signal_variance / self.snr
            noise_std = np.sqrt(self.noise_variance)

            # Set seed for reproducible noise
            torch.manual_seed(self.seed)

            # Apply noise to all splits
            self.Y_train_noisy = self.Y_train_true + torch.randn_like(self.Y_train_true) * noise_std
            self.Y_val_noisy = self.Y_val_true + torch.randn_like(self.Y_val_true) * noise_std
            self.Y_test_noisy = self.Y_test_true + torch.randn_like(self.Y_test_true) * noise_std

        return self

    def compute_comparisons(self) -> 'ExperimentData':
        """
        Generate pairwise comparisons for train and validation sets.

        Uses noisy Y values to generate comparisons (as would be the case in practice).

        Returns:
            self for method chaining.
        """
        # Generate comparisons from noisy values
        self.comparisons_train, self.X_train_pairwise = get_comparisons(
            self.Y_train_noisy, X=self.X_train
        )
        self.comparisons_val, self.X_val_pairwise = get_comparisons(
            self.Y_val_noisy, X=self.X_val
        )

        return self

    def build_dfs(self) -> Dict[str, Dict[str, Any]]:
        """
        Build dictionaries for results storage.

        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing:
                - fold: int (0=train, 1=val, 2=test)
                - X: input data
                - y_true: ground truth
                - y_noisy: noisy observations
        """
        def _build_df(
            fold: int,
            X: torch.Tensor,
            Y_true: torch.Tensor,
            Y_noisy: torch.Tensor
        ) -> Dict[str, Any]:
            return {
                'fold': fold,
                'X': X.numpy().flatten() if self.dimension == 1 else X.numpy().tolist(),
                'y_true': Y_true.numpy().flatten(),
                'y_noisy': Y_noisy.numpy().flatten(),
            }

        return {
            'train': _build_df(0, self.X_train, self.Y_train_true, self.Y_train_noisy),
            'val': _build_df(1, self.X_val, self.Y_val_true, self.Y_val_noisy),
            'test': _build_df(2, self.X_test, self.Y_test_true, self.Y_test_noisy),
        }

    def prepare(self) -> 'ExperimentData':
        """
        Convenience method to run all preparation steps.

        Equivalent to calling sample().noise().compute_comparisons().

        Returns:
            self for method chaining.
        """
        return self.sample().noise().compute_comparisons()

    @property
    def n_train_actual(self) -> int:
        """Number of actual training samples (after validation split)."""
        return len(self.X_train) if self.X_train is not None else 0

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.X_val) if self.X_val is not None else 0
