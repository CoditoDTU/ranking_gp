#!/usr/bin/env python
"""
Visualization runner for GP experiments.

Generates plots from saved experiment results without re-running experiments.
All data is loaded directly from CSV files - no data regeneration needed.

Usage:
    python run_visualization.py                        # Plot latest experiment
    python run_visualization.py --id exp_20260209_123  # Plot specific experiment
    python run_visualization.py --all                  # Plot all experiments
    python run_visualization.py --grid_dir experiments/grid_20260209_123  # Plot grid search
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
from pathlib import Path

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


def visualize_experiment(exp_dir: Path) -> bool:
    """Generate plots for a single experiment directory.

    All data is loaded directly from saved CSV files - no regeneration needed.

    Args:
        exp_dir: Path to the experiment directory.

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

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    generated = 0
    for fn_name in fitness_functions:
        print(f"  Plotting {fn_name}...", end=" ")

        try:
            # Generate plot - all data loaded from CSV, no regeneration needed
            pdf_path = plot_from_saved_results(
                experiment_dir=exp_dir,
                fitness_fn=fn_name,
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


def visualize_grid_search(grid_dir: Path):
    """Generate plots for all experiments in a grid search."""
    runs_dir = grid_dir / "runs"
    if not runs_dir.exists():
        print(f"No runs directory found in {grid_dir}")
        return

    exp_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment runs in grid search")

    generated = 0
    for exp_dir in exp_dirs:
        if visualize_experiment(exp_dir):
            generated += 1

    print(f"\n--- Grid Search Visualization Complete: {generated}/{len(exp_dirs)} experiments plotted ---")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from saved experiment results")
    parser.add_argument("--id", type=str, default=None,
                        help="Experiment ID to plot (e.g., exp_20260209_123)")
    parser.add_argument("--all", action="store_true",
                        help="Plot all experiments")
    parser.add_argument("--grid_dir", type=str, default=None,
                        help="Grid search directory to plot")
    args = parser.parse_args()

    experiments_dir = Path("experiments")

    if args.grid_dir:
        # Plot all experiments in a grid search
        grid_dir = Path(args.grid_dir)
        if not grid_dir.exists():
            print(f"Grid directory not found: {grid_dir}")
            sys.exit(1)
        visualize_grid_search(grid_dir)

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
            if visualize_experiment(exp_dir):
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

        if not visualize_experiment(exp_dir):
            print("Error: No plots could be generated.")
            sys.exit(1)

        print("\n--- Visualization Complete ---")


if __name__ == "__main__":
    main()
