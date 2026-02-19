"""
Abstract base class for GP model wrappers.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import torch
import numpy as np


class BaseModelWrapper(ABC):
    """
    Abstract base class for GP model wrappers.

    Provides a unified interface for ExactGP and PairwiseGP models.
    """

    def __init__(
        self,
        kernel_name: str,
        dimension: int,
        signal_variance: float,
        device: torch.device,
        noise_variance_model: float = None
    ):
        """
        Initialize the model wrapper.

        Args:
            kernel_name: Name of the kernel to use.
            dimension: Input dimensionality.
            snr_model: Expected signal-to-noise ratio for model priors.
            signal_variance: Signal variance from training data.
            device: Torch device for computation.
        """
        self.kernel_name = kernel_name
        self.dimension = dimension
        self.signal_variance = signal_variance
        self.noise_variance_model = noise_variance_model
        self.device = device



        self.model = None
        self.likelihood = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name ('ExactGP' or 'PairwiseGP')."""
        pass

    @abstractmethod
    def build(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        comparisons: Optional[torch.Tensor] = None,
        X_pairwise: Optional[torch.Tensor] = None
    ) -> 'BaseModelWrapper':
        """
        Build the model with training data.

        Args:
            X_train: Training inputs.
            y_train: Training targets.
            comparisons: Pairwise comparison indices (for PairwiseGP).
            X_pairwise: Pairwise input data (for PairwiseGP).

        Returns:
            self for method chaining.
        """
        pass

    @abstractmethod
    def forward(self, X: torch.Tensor) -> Any:
        """
        Forward pass through the model.

        Args:
            X: Input tensor.

        Returns:
            Model output (distribution or posterior).
        """
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with no gradients in eval mode.

        Args:
            X: Input tensor.

        Returns:
            Tuple of (mean, variance) as numpy arrays.
        """
        pass

    @abstractmethod
    def get_lengthscale(self) -> float:
        """Get learned lengthscale."""
        pass

    @abstractmethod
    def get_noise_variance(self) -> float:
        """Get learned/expected noise variance."""
        pass

    def to(self, device: torch.device) -> 'BaseModelWrapper':
        """
        Move model to device.

        Args:
            device: Target device.

        Returns:
            self for method chaining.
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
        if self.likelihood is not None:
            self.likelihood.to(device)
        return self

    def train_mode(self):
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
        if self.likelihood is not None:
            self.likelihood.train()

    def eval_mode(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        if self.likelihood is not None:
            self.likelihood.eval()
