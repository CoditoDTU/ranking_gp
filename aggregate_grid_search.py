#!/usr/bin/env python
"""
Aggregate results from grid search experiments.

Collects best_models.json from each experiment run, creates:
- aggregate_results.csv: All best models from all runs
- best_overall_per_fitness.csv: Summary table of best model per fitness function
- best_overall/: Complete files for overall best per fitness function
- plots/: MLL vs SNR plots

Usage:
    python aggregate_grid_search.py --grid_dir experiments/grid_20260207_143022
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization import (
    plot_mll_vs_snr,
    plot_normalized_mll_vs_snr,
    plot_comparison_heatmap,
)


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

        # Parse experiment name: exp_s{seed}_snr{snr}_{optimizer}
        exp_name = exp_dir.name
        parts = exp_name.split("_")
        # exp_s42_snr10_Adam -> parts = ['exp', 's42', 'snr10', 'Adam']
        seed = int(parts[1][1:])  # 's42' -> 42
        snr = float(parts[2][3:])  # 'snr10' -> 10.0
        optimizer = parts[3]  # 'Adam'

        with open(best_models_file) as f:
            best_models = json.load(f)

        for key, model_data in best_models.items():
            # key is like "ExactGP_ackley"
            gp_type, fitness_fn = key.split("_", 1)

            result = {
                "experiment_id": exp_name,
                "seed": seed,
                "snr_data": snr,
                "optimizer": optimizer,
                "gp_type": gp_type,
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
            all_results.append(result)

    return pd.DataFrame(all_results)


def find_best_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Find the best model per fitness function across all experiments."""
    best_rows = []

    for fitness_fn in df["fitness_fn"].unique():
        fn_df = df[df["fitness_fn"] == fitness_fn]
        # Best = lowest val_mll
        best_idx = fn_df["val_mll"].idxmin()
        best_row = fn_df.loc[best_idx].copy()
        best_rows.append(best_row)

    return pd.DataFrame(best_rows)


def save_best_overall(df: pd.DataFrame, grid_dir: Path):
    """Save complete data for the single best model per fitness function."""
    best_dir = grid_dir / "best_overall"

    for fitness_fn in df["fitness_fn"].unique():
        fn_df = df[df["fitness_fn"] == fitness_fn]
        best_row = fn_df.loc[fn_df["val_mll"].idxmin()]

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

        # Save source reference
        source_info = {
            "experiment_id": best_row["experiment_id"],
            "gp_type": best_row["gp_type"],
            "kernel": best_row["kernel"],
            "seed": int(best_row["seed"]),
            "snr_data": float(best_row["snr_data"]),
            "optimizer": best_row["optimizer"],
            "val_mll": float(best_row["val_mll"]),
            "test_mll": float(best_row["test_mll"]),
            "kendall_tau": float(best_row["kendall_tau"]) if not np.isnan(best_row["kendall_tau"]) else None,
            "spearman": float(best_row["spearman"]) if not np.isnan(best_row["spearman"]) else None,
        }
        with open(dest / "source.json", "w") as f:
            json.dump(source_info, f, indent=2)

        print(f"  Saved best model for {fitness_fn}: {best_row['gp_type']} ({best_row['kernel']})")


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

    # Generate plots
    print("\nGenerating plots...")
    plot_mll_vs_snr(df, grid_dir)
    plot_normalized_mll_vs_snr(df, grid_dir)
    plot_comparison_heatmap(df, grid_dir)

    print("\nAggregation complete!")
    print(f"\nSummary of best models:")
    print(best_df[["fitness_fn", "gp_type", "kernel", "snr_data", "val_mll", "kendall_tau"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    exit(main())
