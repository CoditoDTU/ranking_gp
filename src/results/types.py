"""
Result types for GP ranking experiments.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import numpy as np

from ..config.config import SelectionCriterion


@dataclass
class ModelResult:
    """
    Complete result for a single model configuration.

    Stores all information needed for model selection and analysis.
    """

    # Identifiers
    gp_type: str              # "ExactGP" or "PairwiseGP"
    fitness_fn: str           # "ackley", "levy", etc.
    kernel_name: str          # "matern_5_2", etc.

    # Config
    seed: int
    noise_variance: float
    noise_variance_model: float
    optimizer: str
    lr: float
    training_iters: int

    # Data info
    signal_variance: float
    #noise_variance_data: float      # Actual noise added to data
    lengthscale: float

    # Metrics
    train_mll: float
    val_mll: float
    test_mll: float
    kendall_tau: float
    spearman: float

    # Losses (full history)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)

    # Predictions (optional, populated for best models)
    y_pred_train: Optional[np.ndarray] = None
    y_pred_val: Optional[np.ndarray] = None
    y_pred_test: Optional[np.ndarray] = None
    var_train: Optional[np.ndarray] = None
    var_val: Optional[np.ndarray] = None
    var_test: Optional[np.ndarray] = None

    # Input data and ground truth (optional, for predictions.csv)
    X_train: Optional[np.ndarray] = None
    X_val: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_true_train: Optional[np.ndarray] = None
    y_true_val: Optional[np.ndarray] = None
    y_true_test: Optional[np.ndarray] = None
    y_noisy_train: Optional[np.ndarray] = None
    y_noisy_val: Optional[np.ndarray] = None
    y_noisy_test: Optional[np.ndarray] = None

    def get_criterion_value(self, criterion: SelectionCriterion) -> float:
        """
        Get value for model selection.

        For MLL: returns negative val_mll (lower is better for selection).
        For correlation metrics: returns negative value (higher correlation is better).

        Args:
            criterion: Selection criterion to use.

        Returns:
            Value where lower is better for selection.
        """
        if criterion == SelectionCriterion.VAL_MLL:
            return -self.val_mll  # Negate: higher MLL is better
        elif criterion == SelectionCriterion.KENDALL_TAU:
            return -self.kendall_tau  # Negate: higher tau is better
        elif criterion == SelectionCriterion.SPEARMAN:
            return -self.spearman  # Negate: higher spearman is better
        raise ValueError(f"Unknown criterion: {criterion}")

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding numpy arrays)."""
        return {
            'gp_type': self.gp_type,
            'fitness_fn': self.fitness_fn,
            'kernel_name': self.kernel_name,
            'seed': self.seed,
            'optimizer': self.optimizer,
            'lr': self.lr,
            'training_iters': self.training_iters,
            'noise_variance': self.noise_variance,
            'noise_variance_model': self.noise_variance_model,
            'lengthscale': self.lengthscale,
            'train_mll': self.train_mll,
            'val_mll': self.val_mll,
            'test_mll': self.test_mll,
            'kendall_tau': self.kendall_tau,
            'spearman': self.spearman,
        }


@dataclass
class FailureRecord:
    """
    Record of a failed experiment run.

    Used to track which models failed during training (e.g., PSD matrix errors).
    """
    gp_type: str
    fitness_fn: str
    kernel_name: str
    seed: int
    error_type: str
    error_message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            'gp_type': self.gp_type,
            'fitness_fn': self.fitness_fn,
            'kernel_name': self.kernel_name,
            'seed': self.seed,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
        }
