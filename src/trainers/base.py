"""
Base trainer interface for GP models.

 @abstractmethod: contract, forcing any subclass that inherits from the base class to 
 provide its own concrete implementation of that method. Pairwise and Exact have different methods
 
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import torch
import numpy as np


class BaseTrainer(ABC):
    """Abstract base class for GP trainers."""

    def __init__(
        self,
        kernel_name: str,
        dimension: int,
        training_iters: int,
        lr: float,
        optimizer_name: str,
        device: torch.device,
    ):
        self.kernel_name = kernel_name
        self.dimension = dimension
        self.training_iters = training_iters
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.device = device

        self.model = None
        self.optimizer = None
        self.losses: List[float] = []

    @property
    @abstractmethod 
    def model_name(self) -> str:
        """Return model name (e.g., 'PairwiseGP', 'ExactGP')."""
        pass

    @abstractmethod
    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        comparisons: torch.Tensor = None,
        X_train_pairwise: torch.Tensor = None,
    ) -> List[float]:
        """
        Train the model.

        Args:
            X_train: Training inputs.
            y_train: Training targets.
            comparisons: Pairwise comparisons (for PairwiseGP).
            X_train_pairwise: Pairwise subset of X_train (for PairwiseGP).

        Returns:
            List of loss values per iteration.
        """
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            X: Input points.

        Returns:
            Tuple of (mean predictions, variance).
        """
        pass

    @abstractmethod
    def compute_nll(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        comparisons: torch.Tensor = None,
    ) -> float:
        """Compute negative log likelihood on given data."""
        pass

    @abstractmethod
    def get_lengthscale(self) -> Any:
        """Extract learned lengthscale(s) from the model."""
        pass

    def cleanup(self):
        """Release model resources and free GPU memory."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
