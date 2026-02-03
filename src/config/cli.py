"""
Command-line argument parsing for experiments.
Extracted from module_1.py lines 56-72.
"""
import argparse


def create_experiment_parser() -> argparse.ArgumentParser:
    """Create argument parser for the experiment runner."""
    parser = argparse.ArgumentParser(description="Run GP Experiment")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--noise_type", type=str, default=None,
                        help="Noise type (overrides config)")
    parser.add_argument("--pairwise_training_iters", type=int, default=None,
                        help="PairwiseGP training iterations (overrides config)")
    parser.add_argument("--pairwise_lr", type=float, default=None,
                        help="PairwiseGP learning rate (overrides config)")
    parser.add_argument("--pairwise_optimizer", type=str, default=None,
                        help="PairwiseGP optimizer (overrides config)")
    parser.add_argument("--exact_training_iters", type=int, default=None,
                        help="ExactGP training iterations (overrides config)")
    parser.add_argument("--exact_lr", type=float, default=None,
                        help="ExactGP learning rate (overrides config)")
    parser.add_argument("--exact_optimizer", type=str, default=None,
                        help="ExactGP optimizer (overrides config)")
    parser.add_argument("--noise", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable noise (overrides config). Use --noise or --no-noise")
    parser.add_argument("--fitness_function", type=str, default=None,
                        help="Single fitness function name (overrides config list)")
    parser.add_argument("--nsamples", type=int, default=None,
                        help="Number of training samples (overrides config)")
    parser.add_argument("--g_std", type=float, default=None,
                        help="Gaussian noise std (overrides config noise_params.g_std)")
    parser.add_argument("--val_fraction", type=float, default=None,
                        help="Fraction of training data for validation (0.0-1.0, overrides config)")
    parser.add_argument("--clear_aggregate", action="store_true",
                        help="Clear existing aggregate summary before running")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress terminal output (log to file only)")
    return parser


def create_visualization_parser() -> argparse.ArgumentParser:
    """Create argument parser for the visualization runner."""
    parser = argparse.ArgumentParser(description="Generate plots for GP experiments")
    parser.add_argument("--id", type=str, default=None,
                        help="Specific experiment ID (e.g., 270126_0). Defaults to latest.")
    parser.add_argument("--all", action="store_true",
                        help="Generate plots for all experiments in experiments/")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (fallback for dimension info)")
    return parser
