"""
Grid search visualization functions.

Provides plotting utilities for aggregate grid search results:
- MLL vs SNR plots with error bars
- Normalized MLL plots (z-score per fitness function)
- Comparison heatmaps (ExactGP vs PairwiseGP)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_mll_zscore(df: pd.DataFrame, metric_col: str = "test_mll") -> pd.DataFrame:
    """
    Z-score normalize MLL per fitness function.

    For each fitness function f:
        z_i = (MLL_i - mean_f) / std_f

    This makes all functions comparable on the same scale:
    - z=0 means average performance for that function
    - z=1 means one std better than average
    - z=-1 means one std worse than average

    Args:
        df: DataFrame with experiment results.
        metric_col: Column to normalize (e.g., "test_mll", "val_mll").

    Returns:
        DataFrame with additional column "{metric_col}_normalized".
    """
    df = df.copy()
    normalized_col = f"{metric_col}_normalized"
    df[normalized_col] = np.nan

    for fitness_fn in df["fitness_fn"].unique():
        mask = df["fitness_fn"] == fitness_fn
        values = df.loc[mask, metric_col]
        mean_val = values.mean()
        std_val = values.std()
        if std_val > 0:
            df.loc[mask, normalized_col] = (values - mean_val) / std_val
        else:
            df.loc[mask, normalized_col] = 0.0

    return df


def plot_mll_vs_snr(df: pd.DataFrame, output_dir: Path):
    """
    Plot MLL and Kendall Tau vs SNR with error bars.

    Creates a 1x2 subplot for each GP type showing:
    - Left: Test MLL vs SNR
    - Right: Kendall Tau vs SNR

    Error bars represent std across seeds at each SNR level.

    Args:
        df: DataFrame with grid search results.
        output_dir: Directory to save plots.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for gp_type in ["ExactGP", "PairwiseGP"]:
        gp_df = df[df["gp_type"] == gp_type]
        if gp_df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Test MLL vs SNR
        ax = axes[0]
        for fitness_fn in sorted(gp_df["fitness_fn"].unique()):
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("snr_data")["test_mll"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("SNR (data)")
        ax.set_ylabel("Test MLL")
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{gp_type}: Test MLL vs SNR")
        ax.grid(True, alpha=0.3)

        # Plot 2: Kendall Tau vs SNR
        ax = axes[1]
        for fitness_fn in sorted(gp_df["fitness_fn"].unique()):
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("snr_data")["kendall_tau"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("SNR (data)")
        ax.set_ylabel("Kendall Tau")
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{gp_type}: Kendall Tau vs SNR")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f"mll_vs_snr_{gp_type}.pdf", bbox_inches="tight")
        plt.close()

    print(f"  Saved MLL vs SNR plots to {plots_dir}")


def plot_normalized_mll_vs_snr(df: pd.DataFrame, output_dir: Path):
    """
    Plot z-score normalized MLL vs SNR for cross-function comparison.

    Normalization is done per fitness function, making all functions
    comparable on the same scale (mean=0, std=1 per function).

    Args:
        df: DataFrame with grid search results.
        output_dir: Directory to save plots.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Normalize MLL per fitness function
    df = normalize_mll_zscore(df, "test_mll")

    for gp_type in ["ExactGP", "PairwiseGP"]:
        gp_df = df[df["gp_type"] == gp_type]
        if gp_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for fitness_fn in sorted(gp_df["fitness_fn"].unique()):
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("snr_data")["test_mll_normalized"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("SNR (data)")
        ax.set_ylabel("Normalized Test MLL (z-score)")
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{gp_type}: Normalized Test MLL vs SNR")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")

        plt.tight_layout()
        plt.savefig(plots_dir / f"normalized_mll_vs_snr_{gp_type}.pdf", bbox_inches="tight")
        plt.close()

    print(f"  Saved normalized MLL vs SNR plots to {plots_dir}")


def plot_comparison_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Create heatmap comparing ExactGP vs PairwiseGP performance.

    Creates heatmaps for test_mll and kendall_tau metrics.

    Args:
        df: DataFrame with grid search results.
        output_dir: Directory to save plots.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["test_mll", "kendall_tau"]:
        pivot = df.pivot_table(
            values=metric,
            index="fitness_fn",
            columns="gp_type",
            aggfunc="mean",
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 10))
        cmap = "RdYlGn" if metric == "kendall_tau" else "RdYlGn_r"
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Add values to cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9)

        plt.colorbar(im, ax=ax, label=metric)
        ax.set_title(f"Mean {metric} by GP Type and Fitness Function")
        plt.tight_layout()
        plt.savefig(plots_dir / f"comparison_{metric}.pdf", bbox_inches="tight")
        plt.close()

    print(f"  Saved comparison heatmaps to {plots_dir}")