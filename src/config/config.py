"""
Configuration dataclasses and loader for GP experiments.

New structure with separate Data, Model, Trainer, and Experiment sections.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import os
import yaml


class SelectionCriterion(Enum):
    """Criterion for selecting the best model."""
    VAL_MLL = "val_mll"
    KENDALL_TAU = "kendall_tau"
    SPEARMAN = "spearman"


@dataclass
class DataConfig:
    """Configuration for data generation."""
    fitness_functions: List[str]
    dimension: int
    n_train: int
    n_test: int
    val_fraction: float
    snr: float  # inf = no noise


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    kernels: List[str]
    snr_model: float
    # Note: Lengthscale is learnable by default


@dataclass
class GPTrainerSettings:
    """Settings for a single GP trainer."""
    training_iters: int
    lrs: List[float]  # List of learning rates for grid search
    optimizer: str

    @property
    def lr(self) -> float:
        """Return first LR for backwards compatibility."""
        return self.lrs[0] if self.lrs else 0.01


@dataclass
class TrainerConfig:
    """Configuration for training."""
    exact_gp: GPTrainerSettings
    pairwise_gp: GPTrainerSettings


@dataclass
class ExperimentSettings:
    """Configuration for experiment execution."""
    seed: int
    selection_criterion: SelectionCriterion
    output_dir: str


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
    experiment: ExperimentSettings

    @property
    def has_noise(self) -> bool:
        """Check if noise is enabled (SNR is finite)."""
        return self.data.snr != float('inf')


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file.

    Returns:
        Config object with all settings.
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', config_path
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    # Parse SNR (handle 'inf' string)
    snr_str = raw['data'].get('snr', 10.0)
    if isinstance(snr_str, str) and snr_str.lower() == 'inf':
        snr = float('inf')
    else:
        snr = float(snr_str)

    snr_model_str = raw['model'].get('snr_model', 10.0)
    if isinstance(snr_model_str, str) and snr_model_str.lower() == 'inf':
        snr_model = float('inf')
    else:
        snr_model = float(snr_model_str)

    # Parse LRs (support both 'lr' for single value and 'lrs' for list)
    def parse_lrs(trainer_raw: dict) -> List[float]:
        if 'lrs' in trainer_raw:
            return list(trainer_raw['lrs'])
        elif 'lr' in trainer_raw:
            return [float(trainer_raw['lr'])]
        else:
            return [0.01]  # default

    return Config(
        data=DataConfig(
            fitness_functions=raw['data']['fitness_functions'],
            dimension=raw['data']['dimension'],
            n_train=raw['data']['n_train'],
            n_test=raw['data']['n_test'],
            val_fraction=raw['data']['val_fraction'],
            snr=snr,
        ),
        model=ModelConfig(
            kernels=raw['model']['kernels'],
            snr_model=snr_model,
        ),
        trainer=TrainerConfig(
            exact_gp=GPTrainerSettings(
                training_iters=raw['trainer']['exact_gp']['training_iters'],
                lrs=parse_lrs(raw['trainer']['exact_gp']),
                optimizer=raw['trainer']['exact_gp']['optimizer'],
            ),
            pairwise_gp=GPTrainerSettings(
                training_iters=raw['trainer']['pairwise_gp']['training_iters'],
                lrs=parse_lrs(raw['trainer']['pairwise_gp']),
                optimizer=raw['trainer']['pairwise_gp']['optimizer'],
            ),
        ),
        experiment=ExperimentSettings(
            seed=raw['experiment']['seed'],
            selection_criterion=SelectionCriterion(raw['experiment']['selection_criterion']),
            output_dir=raw['experiment'].get('output_dir', 'experiments/'),
        ),
    )


def load_config_with_overrides(config_path: str, overrides: Dict[str, Any] = None) -> Config:
    """
    Load configuration with CLI overrides.

    Args:
        config_path: Path to config YAML file.
        overrides: Dictionary of CLI overrides (from argparse vars()).

    Returns:
        Config object with overrides applied.
    """
    config = load_config(config_path)

    if overrides is None:
        return config

    # --- Experiment settings ---
    if overrides.get('seed') is not None:
        config.experiment.seed = overrides['seed']

    if overrides.get('output_dir') is not None:
        config.experiment.output_dir = overrides['output_dir']

    if overrides.get('selection_criterion') is not None:
        config.experiment.selection_criterion = SelectionCriterion(overrides['selection_criterion'])

    # --- Data settings ---
    if overrides.get('snr') is not None:
        config.data.snr = float(overrides['snr'])

    if overrides.get('n_train') is not None:
        config.data.n_train = overrides['n_train']

    if overrides.get('n_test') is not None:
        config.data.n_test = overrides['n_test']

    if overrides.get('val_fraction') is not None:
        config.data.val_fraction = overrides['val_fraction']

    if overrides.get('dimension') is not None:
        config.data.dimension = overrides['dimension']

    if overrides.get('fitness_function') is not None:
        # Single fitness function overrides the list
        config.data.fitness_functions = [overrides['fitness_function']]

    # --- Model settings ---
    if overrides.get('snr_model') is not None:
        config.model.snr_model = float(overrides['snr_model'])

    if overrides.get('kernel') is not None:
        # Single kernel overrides the list
        config.model.kernels = [overrides['kernel']]

    # --- Trainer settings (shared) ---
    if overrides.get('optimizer') is not None:
        config.trainer.exact_gp.optimizer = overrides['optimizer']
        config.trainer.pairwise_gp.optimizer = overrides['optimizer']

    if overrides.get('lr') is not None:
        # CLI --lr overrides lrs list with single value (disables LR grid search)
        config.trainer.exact_gp.lrs = [overrides['lr']]
        config.trainer.pairwise_gp.lrs = [overrides['lr']]

    if overrides.get('training_iters') is not None:
        config.trainer.exact_gp.training_iters = overrides['training_iters']
        config.trainer.pairwise_gp.training_iters = overrides['training_iters']

    # --- ExactGP specific ---
    if overrides.get('exact_optimizer') is not None:
        config.trainer.exact_gp.optimizer = overrides['exact_optimizer']

    if overrides.get('exact_lr') is not None:
        config.trainer.exact_gp.lrs = [overrides['exact_lr']]

    if overrides.get('exact_training_iters') is not None:
        config.trainer.exact_gp.training_iters = overrides['exact_training_iters']

    # --- PairwiseGP specific ---
    if overrides.get('pairwise_optimizer') is not None:
        config.trainer.pairwise_gp.optimizer = overrides['pairwise_optimizer']

    if overrides.get('pairwise_lr') is not None:
        config.trainer.pairwise_gp.lrs = [overrides['pairwise_lr']]

    if overrides.get('pairwise_training_iters') is not None:
        config.trainer.pairwise_gp.training_iters = overrides['pairwise_training_iters']

    return config
