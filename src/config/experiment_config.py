"""
Configuration dataclass and loader for GP experiments.
Extracted from module_1.py lines 54-123.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import os
import yaml


@dataclass
class GPSettings:
    """Settings for a GP model (PairwiseGP or ExactGP)."""
    training_iters: int
    lr: float
    optimizer: str  # 'Adam', 'SGD', 'AdamW', 'LBFGS'


@dataclass
class SigmoidSettings:
    """Sigmoid transformation parameters."""
    k: float = 2.0
    x0: float = 5.0


@dataclass
class NoiseParams:
    """Noise parameters for experiments."""
    g_std: float = 1.0
    h_std: float = 0.1
    amplitude_factor: float = 0.5


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    seed: int
    fitness_functions: List[str]
    noise_types: List[str]
    kernel_names: List[str]
    gp_types: List[str]
    noise_params: NoiseParams
    nsamples: int
    n_test_points: int
    noise: bool
    dimension: int
    pairwise_gp: GPSettings
    exact_gp: GPSettings
    sigmoid: SigmoidSettings
    clear_aggregate: bool = False
    val_fraction: float = 0.0


def load_config(config_path: str) -> dict:
    """Load raw config dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_experiment_config(config_path: str, cli_overrides: Dict[str, Any] = None) -> ExperimentConfig:
    """
    Create ExperimentConfig from YAML file with optional CLI overrides.

    Args:
        config_path: Path to config.yaml
        cli_overrides: Dict of CLI argument overrides (seed, noise_type, etc.)

    Returns:
        ExperimentConfig instance
    """
    # Resolve path relative to caller if needed
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', config_path)

    raw = load_config(config_path)
    exp = raw['experiment']

    overrides = cli_overrides or {}

    # Apply overrides: use CLI value if provided, otherwise use config value
    seed = overrides.get('seed') if overrides.get('seed') is not None else exp['seed']
    noise_types = [overrides['noise_type']] if overrides.get('noise_type') is not None else exp['noise_types']
    fitness_functions = [overrides['fitness_function']] if overrides.get('fitness_function') is not None else exp['fitness_functions']
    nsamples = overrides.get('nsamples') if overrides.get('nsamples') is not None else exp['nsamples']
    g_std = overrides.get('g_std') if overrides.get('g_std') is not None else exp['noise_params']['g_std']
    val_fraction = overrides.get('val_fraction') if overrides.get('val_fraction') is not None else exp.get('val_fraction', 0.0)

    pairwise_iters = overrides.get('pairwise_training_iters') if overrides.get('pairwise_training_iters') is not None else exp['pairwise_gp']['training_iters']
    pairwise_lr = overrides.get('pairwise_lr') if overrides.get('pairwise_lr') is not None else exp['pairwise_gp']['lr']
    pairwise_opt = overrides.get('pairwise_optimizer') if overrides.get('pairwise_optimizer') is not None else exp['pairwise_gp']['optimizer']

    exact_iters = overrides.get('exact_training_iters') if overrides.get('exact_training_iters') is not None else exp['exact_gp']['training_iters']
    exact_lr = overrides.get('exact_lr') if overrides.get('exact_lr') is not None else exp['exact_gp']['lr']
    exact_opt = overrides.get('exact_optimizer') if overrides.get('exact_optimizer') is not None else exp['exact_gp']['optimizer']

    return ExperimentConfig(
        seed=seed,
        fitness_functions=fitness_functions,
        noise_types=noise_types,
        kernel_names=exp['kernel_names'],
        gp_types=exp.get('gp_types', ['PairwiseGP', 'ExactGP']),
        noise_params=NoiseParams(
            g_std=g_std,
            h_std=exp['noise_params']['h_std'],
            amplitude_factor=exp['noise_params']['amplitude_factor'],
        ),
        nsamples=nsamples,
        n_test_points=exp.get('n_test_points', 100),
        noise=overrides.get('noise') if overrides.get('noise') is not None else exp['noise'],
        dimension=exp['dimension'],
        pairwise_gp=GPSettings(
            training_iters=pairwise_iters,
            lr=pairwise_lr,
            optimizer=pairwise_opt,
        ),
        exact_gp=GPSettings(
            training_iters=exact_iters,
            lr=exact_lr,
            optimizer=exact_opt,
        ),
        sigmoid=SigmoidSettings(**exp['sigmoid']),
        clear_aggregate=overrides.get('clear_aggregate', False),
        val_fraction=val_fraction,
    )
