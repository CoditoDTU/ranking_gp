"""
Grid search visualization functions.

Provides plotting utilities for aggregate grid search results:
- Combined MLL/Kendall Tau vs SNR plots (3x2 grid)
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
    Plot MLL and Kendall Tau vs SNR in a single 3x2 grid.

    Layout:
        - Columns: ExactGP, PairwiseGP
        - Row 1: Test MLL vs SNR
        - Row 2: Normalized MLL vs SNR (z-score per fitness function)
        - Row 3: Kendall Tau vs SNR

    Error bars represent std across seeds at each SNR level.

    Args:
        df: DataFrame with grid search results.
        output_dir: Directory to save plots.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Normalize MLL per fitness function
    df = normalize_mll_zscore(df, "test_mll")

    gp_types = ["ExactGP", "PairwiseGP"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for col_idx, gp_type in enumerate(gp_types):
        gp_df = df[df["gp_type"] == gp_type]
        if gp_df.empty:
            continue

        fitness_fns = sorted(gp_df["fitness_fn"].unique())

        # Row 1: Test MLL vs SNR
        ax = axes[0, col_idx]
        for fitness_fn in fitness_fns:
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
        ax.set_title(f"{gp_type}: Test MLL vs SNR")
        ax.grid(True, alpha=0.3)
        if col_idx == 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 2: Normalized MLL vs SNR
        ax = axes[1, col_idx]
        for fitness_fn in fitness_fns:
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
        ax.set_title(f"{gp_type}: Normalized MLL vs SNR")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if col_idx == 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 3: Kendall Tau vs SNR
        ax = axes[2, col_idx]
        for fitness_fn in fitness_fns:
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
        ax.set_title(f"{gp_type}: Kendall Tau vs SNR")
        ax.grid(True, alpha=0.3)
        if col_idx == 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_vs_snr.pdf", bbox_inches="tight")
    plt.close()

    print(f"  Saved metrics vs SNR plot to {plots_dir}/metrics_vs_snr.pdf")


def plot_normalized_mll_vs_snr(df: pd.DataFrame, output_dir: Path):
    """
    Deprecated: Now included in plot_mll_vs_snr as row 2.

    Kept for backwards compatibility - does nothing.
    """
    pass


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