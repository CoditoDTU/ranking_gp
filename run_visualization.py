#!/usr/bin/env python
"""
Visualization runner for GP experiments.

Generates plots from saved experiment results without re-running experiments.
Updated to work with new result structure (ModelResult, models/ directory).

Usage:
    python run_visualization.py                        # Plot latest experiment
    python run_visualization.py --id exp_20260209_123  # Plot specific experiment
    python run_visualization.py --all                  # Plot all experiments
    python run_visualization.py --grid_dir experiments/grid_20260209_123  # Plot grid search
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

from src.config import create_visualization_parser, load_config
from src.data import ExperimentData
from src.visualization import plot_from_saved_results


def find_latest_experiment(experiments_dir: Path) -> Path:
    """Find the most recent experiment directory."""
    exp_dirs = sorted(
        [d for d in experiments_dir.iterdir()
         if d.is_dir() and d.name.startswith("exp_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found in {experiments_dir}")
    return exp_dirs[0]


def get_fitness_functions_from_experiment(exp_dir: Path) -> list:
    """Get list of fitness functions from experiment's models directory."""
    models_dir = exp_dir / "models"
    if not models_dir.exists():
        return []

    fitness_fns = set()
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # Parse "ExactGP_ackley" or "PairwiseGP_levy"
            parts = model_dir.name.split("_", 1)
            if len(parts) == 2:
                fitness_fns.add(parts[1])

    return sorted(fitness_fns)


def load_experiment_config(exp_dir: Path, fallback_config_path: Path):
    """Load config from experiment or fallback to default."""
    # Check for saved config in experiment directory
    exp_config = exp_dir / "config.yaml"
    if exp_config.exists():
        return load_config(exp_config)

    # Try config_new.yaml in experiment directory
    exp_config_new = exp_dir / "config_new.yaml"
    if exp_config_new.exists():
        return load_config(exp_config_new)

    # Fallback to provided config
    return load_config(fallback_config_path)


def visualize_experiment(exp_dir: Path, fallback_config_path: Path) -> bool:
    """Generate plots for a single experiment directory.

    Args:
        exp_dir: Path to the experiment directory.
        fallback_config_path: Path to base config.yaml (used if no saved config found).

    Returns:
        True if plots were generated, False if required files were missing.
    """
    print(f"\nProcessing: {exp_dir.name}")

    # Get fitness functions from saved results
    fitness_functions = get_fitness_functions_from_experiment(exp_dir)
    if not fitness_functions:
        print(f"  No model results found in {exp_dir}")
        return False

    print(f"  Found {len(fitness_functions)} fitness functions: {', '.join(fitness_functions)}")

    # Load config
    config = load_experiment_config(exp_dir, fallback_config_path)

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    generated = 0
    for fn_name in fitness_functions:
        print(f"  Plotting {fn_name}...", end=" ")

        # Recreate data for plotting (need X/Y values)
        try:
            data = ExperimentData(
                fitness_fn_name=fn_name,
                dimension=config.data.dimension,
                n_train=config.data.n_train,
                n_test=config.data.n_test,
                val_fraction=config.data.val_fraction,
                snr=config.data.snr,
                seed=config.experiment.seed,
            )
            data.prepare()

            # Generate plot
            pdf_path = plot_from_saved_results(
                experiment_dir=exp_dir,
                fitness_fn=fn_name,
                data=data,
            )

            if pdf_path:
                print(f"saved to {pdf_path.name}")
                generated += 1
            else:
                print("no data")

        except Exception as e:
            print(f"error: {e}")

    print(f"  Done! Generated {generated} plots in: {plots_dir}")
    return generated > 0


def visualize_grid_search(grid_dir: Path, fallback_config_path: Path):
    """Generate plots for all experiments in a grid search."""
    runs_dir = grid_dir / "runs"
    if not runs_dir.exists():
        print(f"No runs directory found in {grid_dir}")
        return

    exp_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment runs in grid search")

    generated = 0
    for exp_dir in exp_dirs:
        if visualize_experiment(exp_dir, fallback_config_path):
            generated += 1

    print(f"\n--- Grid Search Visualization Complete: {generated}/{len(exp_dirs)} experiments plotted ---")


def main():
    parser = create_visualization_parser()
    parser.add_argument("--grid_dir", type=str, default=None,
                        help="Grid search directory to plot")
    args = parser.parse_args()

    config_path = Path(args.config)
    experiments_dir = Path("experiments")

    if args.grid_dir:
        # Plot all experiments in a grid search
        grid_dir = Path(args.grid_dir)
        if not grid_dir.exists():
            print(f"Grid directory not found: {grid_dir}")
            sys.exit(1)
        visualize_grid_search(grid_dir, config_path)

    elif args.all:
        # Plot all experiments
        if not experiments_dir.exists():
            print("No experiments directory found")
            sys.exit(1)

        exp_dirs = sorted([d for d in experiments_dir.iterdir()
                          if d.is_dir() and d.name.startswith("exp_")])

        if not exp_dirs:
            print("No experiment directories found")
            sys.exit(1)

        print(f"Found {len(exp_dirs)} experiments")

        generated = 0
        for exp_dir in exp_dirs:
            if visualize_experiment(exp_dir, config_path):
                generated += 1

        print(f"\n--- Visualization Complete: {generated}/{len(exp_dirs)} experiments plotted ---")

    else:
        # Plot single experiment (latest or specified)
        if args.id:
            exp_dir = experiments_dir / args.id
            if not exp_dir.exists():
                print(f"Experiment not found: {exp_dir}")
                sys.exit(1)
        else:
            try:
                exp_dir = find_latest_experiment(experiments_dir)
            except FileNotFoundError as e:
                print(e)
                sys.exit(1)

        if not visualize_experiment(exp_dir, config_path):
            print("Error: No plots could be generated.")
            sys.exit(1)

        print("\n--- Visualization Complete ---")


if __name__ == "__main__":
    main()