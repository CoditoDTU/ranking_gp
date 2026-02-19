"""
Grid search visualization functions.

Provides plotting utilities for aggregate grid search results:
- Combined MLL/Kendall Tau vs SNR plots (3x2 grid)
- Per-fitness-function plots
- Comparison heatmaps (ExactGP vs PairwiseGP)
"""
from pathlib import Path
from typing import Optional, List

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


def _plot_metrics_grid(
    df: pd.DataFrame,
    output_path: Path,
    fitness_fns: Optional[List[str]] = None,
    title_suffix: str = "",
):
    """
    Plot a 3x2 grid of metrics vs SNR.

    Args:
        df: DataFrame with normalized MLL column.
        output_path: Full path to save the plot.
        fitness_fns: List of fitness functions to include. If None, use all.
        title_suffix: Suffix to add to plot titles.
    """
    gp_types = ["ExactGP", "PairwiseGP"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for col_idx, gp_type in enumerate(gp_types):
        gp_df = df[df["gp_type"] == gp_type]
        if gp_df.empty:
            continue

        if fitness_fns is None:
            plot_fns = sorted(gp_df["fitness_fn"].unique())
        else:
            plot_fns = [fn for fn in fitness_fns if fn in gp_df["fitness_fn"].values]

        # Row 1: Test MLL vs SNR
        ax = axes[0, col_idx]
        for fitness_fn in plot_fns:
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
        ax.set_title(f"{gp_type}: Test MLL vs SNR{title_suffix}")
        ax.grid(True, alpha=0.3)
        if col_idx == 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 2: Normalized MLL vs SNR
        ax = axes[1, col_idx]
        for fitness_fn in plot_fns:
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
        ax.set_title(f"{gp_type}: Normalized MLL vs SNR{title_suffix}")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if col_idx == 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 3: Kendall Tau vs SNR
        ax = axes[2, col_idx]
        for fitness_fn in plot_fns:
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
        ax.set_title(f"{gp_type}: Kendall Tau vs SNR{title_suffix}")
        ax.grid(True, alpha=0.3)
        if col_idx == 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_mll_vs_snr(df: pd.DataFrame, output_dir: Path):
    """
    Plot MLL and Kendall Tau vs SNR in 3x2 grids.

    Creates:
    - metrics_vs_snr.pdf: Combined plot with all fitness functions
    - metrics_vs_snr_{fitness_fn}.pdf: Individual plot per fitness function

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

    # Get all fitness functions
    all_fitness_fns = sorted(df["fitness_fn"].unique())

    # 1. Combined plot with all fitness functions
    _plot_metrics_grid(
        df,
        plots_dir / "metrics_vs_snr.pdf",
        fitness_fns=None,
        title_suffix="",
    )
    print(f"  Saved combined metrics plot to {plots_dir}/metrics_vs_snr.pdf")

    # 2. Individual plots per fitness function
    per_fn_dir = plots_dir / "per_fitness_fn"
    per_fn_dir.mkdir(parents=True, exist_ok=True)

    for fitness_fn in all_fitness_fns:
        _plot_metrics_grid(
            df,
            per_fn_dir / f"metrics_vs_snr_{fitness_fn}.pdf",
            fitness_fns=[fitness_fn],
            title_suffix=f" ({fitness_fn})",
        )

    print(f"  Saved {len(all_fitness_fns)} per-function plots to {per_fn_dir}/")


def plot_normalized_mll_vs_snr(df: pd.DataFrame, output_dir: Path):
    """
    Deprecated: Now included in plot_mll_vs_snr as row 2.

    Kept for backwards compatibility - does nothing.
    """
    pass


def _plot_metrics_grid_ntrain(
    df: pd.DataFrame,
    output_path: Path,
    fitness_fns: Optional[List[str]] = None,
    title_suffix: str = "",
):
    """
    Plot a 3x2 grid of metrics vs n_train.

    Layout:
        - Columns: ExactGP, PairwiseGP
        - Row 1: Test MLL vs n_train
        - Row 2: Normalized MLL vs n_train (z-score per fitness function)
        - Row 3: Kendall Tau vs n_train

    Args:
        df: DataFrame with normalized MLL column.
        output_path: Full path to save the plot.
        fitness_fns: List of fitness functions to include. If None, use all.
        title_suffix: Suffix to add to plot titles.
    """
    gp_types = ["ExactGP", "PairwiseGP"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for col_idx, gp_type in enumerate(gp_types):
        gp_df = df[df["gp_type"] == gp_type]
        if gp_df.empty:
            continue

        if fitness_fns is None:
            plot_fns = sorted(gp_df["fitness_fn"].unique())
        else:
            plot_fns = [fn for fn in fitness_fns if fn in gp_df["fitness_fn"].values]

        # Row 1: Test MLL vs n_train
        ax = axes[0, col_idx]
        for fitness_fn in plot_fns:
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("n_train")["test_mll"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("$n_{train}$")
        ax.set_ylabel("Test MLL")
        ax.set_title(f"{gp_type}: Test MLL vs $n_{{train}}${title_suffix}")
        ax.grid(True, alpha=0.3)
        if col_idx == 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 2: Normalized MLL vs n_train
        ax = axes[1, col_idx]
        for fitness_fn in plot_fns:
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("n_train")["test_mll_normalized"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("$n_{train}$")
        ax.set_ylabel("Normalized Test MLL (z-score)")
        ax.set_title(f"{gp_type}: Normalized MLL vs $n_{{train}}${title_suffix}")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if col_idx == 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 3: Kendall Tau vs n_train
        ax = axes[2, col_idx]
        for fitness_fn in plot_fns:
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("n_train")["kendall_tau"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("$n_{train}$")
        ax.set_ylabel("Kendall Tau")
        ax.set_title(f"{gp_type}: Kendall Tau vs $n_{{train}}${title_suffix}")
        ax.grid(True, alpha=0.3)
        if col_idx == 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_mll_vs_ntrain(df: pd.DataFrame, output_dir: Path):
    """
    Plot MLL and Kendall Tau vs n_train in 3x2 grids.

    Creates:
    - metrics_vs_ntrain.pdf: Combined plot with all fitness functions
    - metrics_vs_ntrain_{fitness_fn}.pdf: Individual plot per fitness function

    Layout:
        - Columns: ExactGP, PairwiseGP
        - Row 1: Test MLL vs n_train
        - Row 2: Normalized MLL vs n_train (z-score per fitness function)
        - Row 3: Kendall Tau vs n_train

    Error bars represent std across seeds at each n_train level.

    Args:
        df: DataFrame with grid search results.
        output_dir: Directory to save plots.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Normalize MLL per fitness function
    df = normalize_mll_zscore(df, "test_mll")

    # Get all fitness functions
    all_fitness_fns = sorted(df["fitness_fn"].unique())

    # 1. Combined plot with all fitness functions
    _plot_metrics_grid_ntrain(
        df,
        plots_dir / "metrics_vs_ntrain.pdf",
        fitness_fns=None,
        title_suffix="",
    )
    print(f"  Saved combined metrics plot to {plots_dir}/metrics_vs_ntrain.pdf")

    # 2. Individual plots per fitness function
    per_fn_dir = plots_dir / "per_fitness_fn"
    per_fn_dir.mkdir(parents=True, exist_ok=True)

    for fitness_fn in all_fitness_fns:
        _plot_metrics_grid_ntrain(
            df,
            per_fn_dir / f"metrics_vs_ntrain_{fitness_fn}.pdf",
            fitness_fns=[fitness_fn],
            title_suffix=f" ({fitness_fn})",
        )

    print(f"  Saved {len(all_fitness_fns)} per-function plots to {per_fn_dir}/")


def _plot_metrics_grid_noise_variance(
    df: pd.DataFrame,
    output_path: Path,
    fitness_fns: Optional[List[str]] = None,
    title_suffix: str = "",
):
    """
    Plot a 3xN grid of metrics vs noise_variance.

    Layout:
        - Columns: One per GP type (ExactGP variants + PairwiseGP)
        - Row 1: Test MLL vs noise_variance
        - Row 2: Normalized MLL vs noise_variance (z-score per fitness function)
        - Row 3: Kendall Tau vs noise_variance

    Args:
        df: DataFrame with normalized MLL column.
        output_path: Full path to save the plot.
        fitness_fns: List of fitness functions to include. If None, use all.
        title_suffix: Suffix to add to plot titles.
    """
    # Get all unique GP types, sorted with ExactGP variants first, then PairwiseGP
    all_gp_types = sorted(df["gp_type"].unique())
    exactgp_types = [t for t in all_gp_types if t.startswith("ExactGP")]
    pairwise_types = [t for t in all_gp_types if t.startswith("PairwiseGP")]
    gp_types = exactgp_types + pairwise_types

    n_cols = len(gp_types)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, gp_type in enumerate(gp_types):
        gp_df = df[df["gp_type"] == gp_type]
        if gp_df.empty:
            continue

        if fitness_fns is None:
            plot_fns = sorted(gp_df["fitness_fn"].unique())
        else:
            plot_fns = [fn for fn in fitness_fns if fn in gp_df["fitness_fn"].values]

        # Row 1: Test MLL vs noise_variance
        ax = axes[0, col_idx]
        for fitness_fn in plot_fns:
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("noise_variance")["test_mll"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("Noise Variance (σ²)")
        ax.set_ylabel("Test MLL")
        ax.set_title(f"{gp_type}: Test MLL{title_suffix}")
        ax.grid(True, alpha=0.3)
        if col_idx == n_cols - 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 2: Normalized MLL vs noise_variance
        ax = axes[1, col_idx]
        for fitness_fn in plot_fns:
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("noise_variance")["test_mll_normalized"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("Noise Variance (σ²)")
        ax.set_ylabel("Normalized Test MLL (z-score)")
        ax.set_title(f"{gp_type}: Normalized MLL{title_suffix}")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if col_idx == n_cols - 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 3: Kendall Tau vs noise_variance
        ax = axes[2, col_idx]
        for fitness_fn in plot_fns:
            subset = gp_df[gp_df["fitness_fn"] == fitness_fn]
            stats = subset.groupby("noise_variance")["kendall_tau"].agg(["mean", "std"])

            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                capthick=1,
                label=fitness_fn,
            )

        ax.set_xlabel("Noise Variance (σ²)")
        ax.set_ylabel("Kendall Tau")
        ax.set_title(f"{gp_type}: Kendall Tau{title_suffix}")
        ax.grid(True, alpha=0.3)
        if col_idx == n_cols - 1 and len(plot_fns) > 1:
            ax.legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_mll_vs_noise_variance(df: pd.DataFrame, output_dir: Path):
    """
    Plot MLL and Kendall Tau vs noise_variance in 3x2 grids.

    Creates:
    - metrics_vs_noise_variance.pdf: Combined plot with all fitness functions
    - metrics_vs_noise_variance_{fitness_fn}.pdf: Individual plot per fitness function

    Layout:
        - Columns: ExactGP, PairwiseGP
        - Row 1: Test MLL vs noise_variance
        - Row 2: Normalized MLL vs noise_variance (z-score per fitness function)
        - Row 3: Kendall Tau vs noise_variance

    Error bars represent std across seeds at each noise_variance level.

    Args:
        df: DataFrame with grid search results.
        output_dir: Directory to save plots.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Normalize MLL per fitness function
    df = normalize_mll_zscore(df, "test_mll")

    # Get all fitness functions
    all_fitness_fns = sorted(df["fitness_fn"].unique())

    # 1. Combined plot with all fitness functions
    _plot_metrics_grid_noise_variance(
        df,
        plots_dir / "metrics_vs_noise_variance.pdf",
        fitness_fns=None,
        title_suffix="",
    )
    print(f"  Saved combined metrics plot to {plots_dir}/metrics_vs_noise_variance.pdf")

    # 2. Individual plots per fitness function
    per_fn_dir = plots_dir / "per_fitness_fn"
    per_fn_dir.mkdir(parents=True, exist_ok=True)

    for fitness_fn in all_fitness_fns:
        _plot_metrics_grid_noise_variance(
            df,
            per_fn_dir / f"metrics_vs_noise_variance_{fitness_fn}.pdf",
            fitness_fns=[fitness_fn],
            title_suffix=f" ({fitness_fn})",
        )

    print(f"  Saved {len(all_fitness_fns)} per-function plots to {per_fn_dir}/")


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