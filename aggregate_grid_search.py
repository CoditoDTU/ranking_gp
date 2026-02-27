#!/usr/bin/env python
"""
Aggregate results from grid search experiments.

Collects best_models.json from each experiment run, creates:
- aggregate_results.csv: All best models from all runs
- best_overall_per_fitness.csv: Summary table of best model per fitness function
- best_overall/: Complete files for overall best per fitness function
- plots/: Performance plots vs grid search variables

Supports multiple experiment naming formats:
- exp_{seed}_sigma_{noise_variance}_{optimizer}  (noise_variance grid search)
- exp_{seed}_n_{n_train}                         (n_train grid search)
- exp_{seed}_sigma_{noise_variance}_n_{n_train}  (multi-variable grid search)

Usage:
    python aggregate_grid_search.py --grid_dir experiments/grid_20260207_143022
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visualization import (
    plot_mll_vs_ntrain,
    plot_mll_vs_noise_variance,
    plot_comparison_heatmap,
)


def parse_experiment_name(exp_name: str) -> dict:
    """
    Parse experiment name to extract grid search variables.

    Supports formats like:
    - exp_0_sigma_0.1_Adam -> {seed: 0, noise_variance: 0.1, optimizer: 'Adam'}
    - exp_0_n_30 -> {seed: 0, n_train: 30}
    - exp_1_sigma_0.5_n_50 -> {seed: 1, noise_variance: 0.5, n_train: 50}
    - exp_s42_n256 -> {seed: 42, n_train: 256}  (n_train grid search format)
    """
    result = {}

    # Extract seed: exp_{number}_ or exp_s{number}_ (supports both formats)
    seed_match = re.search(r'^exp_s?(\d+)_', exp_name)
    if seed_match:
        result['seed'] = int(seed_match.group(1))

    # Extract noise_variance: sigma_{number}
    sigma_match = re.search(r'_sigma_([\d.]+)', exp_name)
    if sigma_match:
        result['noise_variance'] = float(sigma_match.group(1))

    # Extract n_train: n_{number} or n{number} (underscore after n is optional)
    ntrain_match = re.search(r'_n_?(\d+)', exp_name)
    if ntrain_match:
        result['n_train'] = int(ntrain_match.group(1))

    # Extract optimizer: last part if it's a known optimizer name
    parts = exp_name.split('_')
    if parts[-1] in ['Adam', 'SGD', 'AdamW', 'RMSprop']:
        result['optimizer'] = parts[-1]

    return result


def collect_results(grid_dir: Path) -> pd.DataFrame:
    """Collect all best_models.json files from experiment runs."""
    runs_dir = grid_dir / "runs"
    all_results = []

    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        best_models_file = exp_dir / "best_models.json"
        if not best_models_file.exists():
            print(f"Warning: No best_models.json in {exp_dir.name}")
            continue

        # Parse experiment name dynamically
        exp_name = exp_dir.name
        exp_vars = parse_experiment_name(exp_name)

        with open(best_models_file) as f:
            best_models = json.load(f)

        for key, model_data in best_models.items():
            # key is like "ExactGP_ackley" or "ExactGP_0.5x_ackley" or "PairwiseGP_ackley"
            exactgp_match = re.match(r'(ExactGP_[\d.]+x)_(.+)', key)
            if exactgp_match:
                gp_type = exactgp_match.group(1)
                gp_type_base = "ExactGP"
                fitness_fn = exactgp_match.group(2)
            else:
                gp_type, fitness_fn = key.split("_", 1)
                gp_type_base = gp_type

            result = {
                "experiment_id": exp_name,
                "gp_type": gp_type,
                "gp_type_base": gp_type_base,
                "fitness_fn": fitness_fn,
                "kernel": model_data.get("kernel_name", "unknown"),
                "val_mll": model_data.get("val_mll", np.nan),
                "test_mll": model_data.get("test_mll", np.nan),
                "kendall_tau": model_data.get("kendall_tau", np.nan),
                "spearman": model_data.get("spearman", np.nan),
                "train_mll": model_data.get("train_mll", np.nan),
                "lengthscale": model_data.get("lengthscale", np.nan),
                "signal_variance": model_data.get("signal_variance", np.nan),
                "noise_variance_data": model_data.get("noise_variance_data", np.nan),
                "noise_variance_model": model_data.get("noise_variance_model", np.nan),
                "lr": model_data.get("lr", np.nan),
                "training_iters": model_data.get("training_iters", 0),
            }
            # Add parsed experiment variables
            result.update(exp_vars)
            all_results.append(result)

    return pd.DataFrame(all_results)


def detect_grid_variables(df: pd.DataFrame) -> list:
    """Detect which variables are being swept in the grid search."""
    grid_vars = []

    # Check for noise_variance sweep
    if 'noise_variance' in df.columns and df['noise_variance'].nunique() > 1:
        grid_vars.append('noise_variance')

    # Check for n_train sweep
    if 'n_train' in df.columns and df['n_train'].nunique() > 1:
        grid_vars.append('n_train')

    # Check for optimizer sweep
    if 'optimizer' in df.columns and df['optimizer'].nunique() > 1:
        grid_vars.append('optimizer')

    return grid_vars


def find_best_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Find the best model per fitness function across all experiments."""
    best_rows = []

    for fitness_fn in df["fitness_fn"].unique():
        fn_df = df[df["fitness_fn"] == fitness_fn]
        # Best = highest kendall_tau
        best_idx = fn_df["kendall_tau"].idxmax()
        best_row = fn_df.loc[best_idx].copy()
        best_rows.append(best_row)

    return pd.DataFrame(best_rows)


def save_best_overall(df: pd.DataFrame, grid_dir: Path):
    """Save complete data for the single best model per fitness function."""
    best_dir = grid_dir / "best_overall"

    for fitness_fn in df["fitness_fn"].unique():
        fn_df = df[df["fitness_fn"] == fitness_fn]
        best_row = fn_df.loc[fn_df["kendall_tau"].idxmax()]

        # Find the source experiment directory
        source_exp = grid_dir / "runs" / best_row["experiment_id"]
        source_model = source_exp / "models" / f"{best_row['gp_type']}_{fitness_fn}"

        # Destination directory
        dest = best_dir / fitness_fn
        dest.mkdir(parents=True, exist_ok=True)

        # Copy all model files if they exist
        files_to_copy = ["losses.json", "hyperparams.json", "metrics.json", "predictions.csv"]
        for fname in files_to_copy:
            src_file = source_model / fname
            if src_file.exists():
                shutil.copy(src_file, dest / fname)
            else:
                print(f"Warning: {src_file} not found")

        # Save source reference - include all available experiment variables
        source_info = {
            "experiment_id": best_row["experiment_id"],
            "gp_type": best_row["gp_type"],
            "kernel": best_row["kernel"],
            "val_mll": float(best_row["val_mll"]),
            "test_mll": float(best_row["test_mll"]),
            "kendall_tau": float(best_row["kendall_tau"]) if not np.isnan(best_row["kendall_tau"]) else None,
            "spearman": float(best_row["spearman"]) if not np.isnan(best_row["spearman"]) else None,
        }
        # Add optional fields if present
        if 'seed' in best_row:
            source_info['seed'] = int(best_row['seed'])
        if 'noise_variance' in best_row and not np.isnan(best_row['noise_variance']):
            source_info['noise_variance'] = float(best_row['noise_variance'])
        if 'n_train' in best_row:
            source_info['n_train'] = int(best_row['n_train'])
        if 'optimizer' in best_row:
            source_info['optimizer'] = best_row['optimizer']

        with open(dest / "source.json", "w") as f:
            json.dump(source_info, f, indent=2)

        print(f"  Saved best model for {fitness_fn}: {best_row['gp_type']} ({best_row['kernel']})")


def plot_performance_vs_variable(df: pd.DataFrame, grid_dir: Path, x_var: str,
                                  y_var: str = 'kendall_tau', use_log_x: bool = False):
    """
    Generic plot of performance metric vs grid search variable.

    Args:
        df: DataFrame with results
        grid_dir: Output directory
        x_var: Column name for x-axis (e.g., 'noise_variance', 'n_train')
        y_var: Column name for y-axis metric (default: 'kendall_tau')
        use_log_x: Whether to use log scale for x-axis
    """
    plot_dir = grid_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Compute mean and std by x_var and GP type
    stats = df.groupby([x_var, 'gp_type'])[y_var].agg(['mean', 'std']).reset_index()

    exact = stats[stats['gp_type'] == 'ExactGP'].copy()
    pair = stats[stats['gp_type'] == 'PairwiseGP'].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(exact[x_var], exact['mean'], yerr=exact['std'],
                fmt='o-', label='ExactGP', capsize=3, markersize=6)
    ax.errorbar(pair[x_var], pair['mean'], yerr=pair['std'],
                fmt='s-', label='PairwiseGP', capsize=3, markersize=6)

    if use_log_x:
        ax.set_xscale('log')

    # Format labels
    x_labels = {
        'noise_variance': 'Noise Variance (σ²)',
        'n_train': 'Training Set Size (n_train)',
    }
    y_labels = {
        'kendall_tau': "Kendall's τ",
        'spearman': "Spearman's ρ",
        'val_mll': 'Validation MLL',
    }

    ax.set_xlabel(x_labels.get(x_var, x_var), fontsize=12)
    ax.set_ylabel(y_labels.get(y_var, y_var), fontsize=12)
    ax.set_title(f"{y_labels.get(y_var, y_var)} vs {x_labels.get(x_var, x_var)}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if y_var in ['kendall_tau', 'spearman']:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(plot_dir / f"{y_var}_vs_{x_var}.pdf")
    plt.close()
    print(f"  Saved {y_var}_vs_{x_var} plots")


def plot_performance_by_kernel(df: pd.DataFrame, grid_dir: Path, x_var: str,
                                y_var: str = 'kendall_tau', use_log_x: bool = False):
    """Plot performance vs variable, broken down by kernel."""
    plot_dir = grid_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for gp_type in df['gp_type'].unique():
        gp_df = df[df['gp_type'] == gp_type].copy()

        stats = gp_df.groupby([x_var, 'kernel'])[y_var].agg(['mean', 'std']).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        for kernel in stats['kernel'].unique():
            kernel_data = stats[stats['kernel'] == kernel]
            ax.errorbar(kernel_data[x_var], kernel_data['mean'], yerr=kernel_data['std'],
                        fmt='o-', label=kernel, capsize=3, markersize=5)

        if use_log_x:
            ax.set_xscale('log')

        x_labels = {'noise_variance': 'Noise Variance (σ²)', 'n_train': 'Training Set Size'}
        y_labels = {'kendall_tau': "Kendall's τ", 'spearman': "Spearman's ρ"}

        ax.set_xlabel(x_labels.get(x_var, x_var), fontsize=12)
        ax.set_ylabel(y_labels.get(y_var, y_var), fontsize=12)
        ax.set_title(f"{gp_type}: {y_labels.get(y_var, y_var)} vs {x_labels.get(x_var, x_var)}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if y_var in ['kendall_tau', 'spearman']:
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(plot_dir / f"{y_var}_vs_{x_var}_{gp_type}.pdf")
        plt.close()

    print(f"  Saved {y_var}_vs_{x_var} by kernel plots")


def plot_heatmap(df: pd.DataFrame, grid_dir: Path, x_var: str, y_var: str = 'kendall_tau'):
    """Plot heatmap of performance metric by x_var and kernel."""
    plot_dir = grid_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for gp_type in df['gp_type'].unique():
        gp_df = df[df['gp_type'] == gp_type]
        pivot = gp_df.pivot_table(values=y_var, index='kernel', columns=x_var, aggfunc='mean')

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Format column labels
        col_labels = []
        for col in pivot.columns:
            if isinstance(col, float) and col == int(col):
                col_labels.append(str(int(col)))
            else:
                col_labels.append(str(col))

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        x_labels = {'noise_variance': 'Noise Variance (σ²)', 'n_train': 'n_train'}
        y_labels = {'kendall_tau': "Kendall's τ", 'spearman': "Spearman's ρ"}

        ax.set_xlabel(x_labels.get(x_var, x_var), fontsize=12)
        ax.set_ylabel('Kernel', fontsize=12)
        ax.set_title(f"{gp_type}: {y_labels.get(y_var, y_var)} by {x_labels.get(x_var, x_var)} and Kernel", fontsize=14)

        plt.colorbar(im, ax=ax, label=y_labels.get(y_var, y_var))

        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

        plt.tight_layout()
        plt.savefig(plot_dir / f"{y_var}_heatmap_{x_var}_{gp_type}.pdf")
        plt.close()

    print(f"  Saved {y_var} heatmap plots")


def main():
    parser = argparse.ArgumentParser(description="Aggregate grid search results")
    parser.add_argument("--grid_dir", type=str, required=True, help="Grid search directory")
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    if not grid_dir.exists():
        print(f"Error: Grid directory not found: {grid_dir}")
        return 1

    print(f"Aggregating results from: {grid_dir}")

    # Collect all results
    print("\nCollecting results from experiment runs...")
    df = collect_results(grid_dir)

    if df.empty:
        print("Error: No results found!")
        return 1

    print(f"  Found {len(df)} results across {df['experiment_id'].nunique()} experiments")

    # Detect grid search variables
    grid_vars = detect_grid_variables(df)
    print(f"  Detected grid search variables: {grid_vars}")

    # Save aggregate results
    aggregate_file = grid_dir / "aggregate_results.csv"
    df.to_csv(aggregate_file, index=False)
    print(f"\nSaved aggregate results to: {aggregate_file}")

    # Find and save best overall per fitness function
    print("\nFinding best models per fitness function...")
    best_df = find_best_overall(df)
    best_file = grid_dir / "best_overall_per_fitness.csv"
    best_df.to_csv(best_file, index=False)
    print(f"Saved best models summary to: {best_file}")

    # Copy complete data for best models
    print("\nCopying complete data for best models...")
    save_best_overall(df, grid_dir)

    # Generate plots based on detected variables
    print("\nGenerating plots...")

    for x_var in grid_vars:
        # Main performance plots
        plot_performance_vs_variable(df, grid_dir, x_var, 'kendall_tau')
        plot_performance_vs_variable(df, grid_dir, x_var, 'spearman')

        # By kernel breakdown
        plot_performance_by_kernel(df, grid_dir, x_var, 'kendall_tau')

        # Heatmaps
        plot_heatmap(df, grid_dir, x_var, 'kendall_tau')

    # noise_variance-specific plots (3x2 grid with MLL, normalized MLL, Kendall Tau)
    if 'noise_variance' in grid_vars:
        try:
            plot_mll_vs_noise_variance(df, grid_dir)
            plot_comparison_heatmap(df, grid_dir)
        except Exception as e:
            print(f"  Warning: noise_variance plots failed: {e}")

    # n_train-specific plots (3x2 grid with MLL, normalized MLL, Kendall Tau)
    if 'n_train' in grid_vars:
        try:
            plot_mll_vs_ntrain(df, grid_dir)
            plot_comparison_heatmap(df, grid_dir)
        except Exception as e:
            print(f"  Warning: n_train plots failed: {e}")

    print("\nAggregation complete!")

    # Print summary with available columns
    summary_cols = ["fitness_fn", "gp_type", "kernel"]
    summary_cols.extend([v for v in grid_vars if v in best_df.columns])
    summary_cols.extend(["kendall_tau"])
    available_cols = [c for c in summary_cols if c in best_df.columns]

    print(f"\nSummary of best models:")
    print(best_df[available_cols].to_string(index=False))

    # Print performance summary by grid variable
    for x_var in grid_vars:
        print(f"\nPerformance by {x_var}:")
        summary = df.groupby([x_var, 'gp_type'])['kendall_tau'].agg(['mean', 'std'])
        print(summary.round(3).to_string())

    return 0


if __name__ == "__main__":
    exit(main())
