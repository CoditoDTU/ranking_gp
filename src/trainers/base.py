"""
Base trainer interface for GP models.

Simplified to work with ExperimentData and model wrappers.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List
import torch

from ..models.base import BaseModelWrapper
from ..data.dataset import ExperimentData


class BaseTrainer(ABC):
    """
    Abstract base class for GP trainers.

    Trainers handle the training loop and loss computation.
    Models are built externally using model wrappers.
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        training_iters: int,
        lr: float,
        optimizer_name: str,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model wrapper (ExactGPModel or PairwiseGPModel).
            training_iters: Number of training iterations.
            lr: Learning rate.
            optimizer_name: Name of optimizer ('Adam', 'SGD', 'LBFGS').
        """
        self.model = model
        self.training_iters = training_iters
        self.lr = lr
        self.optimizer_name = optimizer_name

        self.optimizer = None
        self.losses: List[float] = []
        self.val_losses: List[float] = []

    @property
    def model_name(self) -> str:
        """Return model name from wrapper."""
        return self.model.name

    @abstractmethod
    def train(self, data: ExperimentData) -> Tuple[List[float], List[float]]:
        """
        Train the model using ExperimentData.

        Args:
            data: ExperimentData object with all splits prepared.

        Returns:
            Tuple of (train_losses, val_losses) per iteration.
        """
        pass

    @abstractmethod
    def compute_mll(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        comparisons: torch.Tensor = None,
        X_pairwise: torch.Tensor = None,
    ) -> float:
        """
        Compute marginal log likelihood on given data.

        Args:
            X: Input tensor.
            y: Target tensor.
            comparisons: Pairwise comparison indices (for PairwiseGP).
            X_pairwise: Pairwise input data (for PairwiseGP).

        Returns:
            MLL value (higher is better, unlike NLL).
        """
        pass

    def cleanup(self):
        """Release resources and free GPU memory."""
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()