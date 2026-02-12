"""
Command-line argument parsing for experiments.

Updated for new config structure with SNR-based noise.
"""
import argparse


def create_experiment_parser() -> argparse.ArgumentParser:
    """Create argument parser for the experiment runner."""
    parser = argparse.ArgumentParser(description="Run GP Ranking Experiment")

    # Config file
    parser.add_argument("--config", type=str, default="configs/config_new.yaml",
                        help="Path to config file")

    # Experiment settings
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")

    # Data settings
    parser.add_argument("--snr", type=float, default=None,
                        help="Data SNR (overrides config). Use 'inf' for no noise.")
    parser.add_argument("--n_train", type=int, default=None,
                        help="Number of training samples (overrides config)")
    parser.add_argument("--n_test", type=int, default=None,
                        help="Number of test samples (overrides config)")
    parser.add_argument("--val_fraction", type=float, default=None,
                        help="Fraction of training data for validation (overrides config)")
    parser.add_argument("--fitness_function", type=str, default=None,
                        help="Single fitness function name (overrides config list)")
    parser.add_argument("--dimension", type=int, default=None,
                        help="Input dimension (overrides config)")

    # Model settings
    parser.add_argument("--snr_model", type=float, default=None,
                        help="Model SNR for priors (overrides config)")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Single kernel name (overrides config list)")

    # Trainer settings - shared
    parser.add_argument("--optimizer", type=str, default=None,
                        help="Optimizer for both GP types (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for both GP types (overrides config)")
    parser.add_argument("--training_iters", type=int, default=None,
                        help="Training iterations for both GP types (overrides config)")

    # Trainer settings - ExactGP specific
    parser.add_argument("--exact_training_iters", type=int, default=None,
                        help="ExactGP training iterations (overrides config)")
    parser.add_argument("--exact_lr", type=float, default=None,
                        help="ExactGP learning rate (overrides config)")
    parser.add_argument("--exact_optimizer", type=str, default=None,
                        help="ExactGP optimizer (overrides config)")

    # Trainer settings - PairwiseGP specific
    parser.add_argument("--pairwise_training_iters", type=int, default=None,
                        help="PairwiseGP training iterations (overrides config)")
    parser.add_argument("--pairwise_lr", type=float, default=None,
                        help="PairwiseGP learning rate (overrides config)")
    parser.add_argument("--pairwise_optimizer", type=str, default=None,
                        help="PairwiseGP optimizer (overrides config)")

    # Selection criterion
    parser.add_argument("--selection_criterion", type=str, default=None,
                        choices=["val_mll", "kendall_tau", "spearman"],
                        help="Model selection criterion (overrides config)")

    # Output control
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")

    return parser


def create_visualization_parser() -> argparse.ArgumentParser:
    """Create argument parser for the visualization runner."""
    parser = argparse.ArgumentParser(description="Generate plots for GP experiments")
    parser.add_argument("--id", type=str, default=None,
                        help="Specific experiment ID (e.g., exp_20260207_143022). Defaults to latest.")
    parser.add_argument("--all", action="store_true",
                        help="Generate plots for all experiments in experiments/")
    parser.add_argument("--config", type=str, default="configs/config_new.yaml",
                        help="Path to config file (fallback for dimension info)")
    return parser
