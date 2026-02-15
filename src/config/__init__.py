"""
Configuration module for GP experiments.

Provides both old and new config systems for compatibility.
"""
# New config system
from .config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainerConfig,
    GPTrainerSettings,
    ExperimentSettings,
    SelectionCriterion,
    load_config,
    load_config_with_overrides,
    config_to_dict,
    save_config,
)

# CLI parsers
from .cli import create_experiment_parser, create_visualization_parser

# Old config system (for backwards compatibility)
from .experiment_config import ExperimentConfig, create_experiment_config

__all__ = [
    # New config
    'Config',
    'DataConfig',
    'ModelConfig',
    'TrainerConfig',
    'GPTrainerSettings',
    'ExperimentSettings',
    'SelectionCriterion',
    'load_config',
    'load_config_with_overrides',
    'config_to_dict',
    'save_config',
    # CLI
    'create_experiment_parser',
    'create_visualization_parser',
    # Old config (backwards compat)
    'ExperimentConfig',
    'create_experiment_config',
]
