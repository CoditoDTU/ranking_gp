"""
Visualization plots for GP ranking experiments.

Updated for new result types (ModelResult, ExperimentData).
"""
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..results.types import ModelResult
from ..data.dataset import ExperimentData


def plot_fitness_function_grid(
    fitness_fn: str,
    data: ExperimentData,
    exact_result: Optional[ModelResult],
    pairwise_result: Optional[ModelResult],
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate a 4x2 grid plot for a fitness function showing best models.

    Layout:
        Row 0: Training Loss   (ExactGP | PairwiseGP)
        Row 1: Validation Loss (ExactGP | PairwiseGP)
        Row 2: Function Fit    (ExactGP | PairwiseGP)
        Row 3: Monotonicity    (ExactGP | PairwiseGP)

    Args:
        fitness_fn: Name of the fitness function.
        data: ExperimentData with X/Y arrays.
        exact_result: ModelResult for best ExactGP (or None).
        pairwise_result: ModelResult for best PairwiseGP (or None).
        output_dir: Directory to save the plot.

    Returns:
        Path to saved PDF, or None if no results.
    """
    if exact_result is None and pairwise_result is None:
        return None

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    # Build title
    kernels = []
    if exact_result:
        kernels.append(f"ExactGP: {exact_result.kernel_name}")
    if pairwise_result:
        kernels.append(f"PairwiseGP: {pairwise_result.kernel_name}")
    fig.suptitle(f'{fitness_fn}\n{" | ".join(kernels)}', fontsize=14, fontweight='bold')

    # Prepare test data for plots
    X_test = data.X_test.numpy()
    Y_test = data.Y_test_true.numpy()
    X_train = data.X_train.numpy()
    Y_train = data.Y_train_noisy.numpy()

    # For 1D, sort for proper line plots
    if data.dimension == 1:
        X_test_flat = X_test.flatten()
        sort_idx = np.argsort(X_test_flat)
        X_test_sorted = X_test_flat[sort_idx]
        Y_test_sorted = Y_test.flatten()[sort_idx]
        X_train_flat = X_train.flatten()
        Y_train_flat = Y_train.flatten()
    else:
        sort_idx = None
        X_test_sorted = Y_test_sorted = None
        X_train_flat = Y_train_flat = None

    # ===== ROW 0: Training Loss =====
    _plot_loss(axes[0, 0], exact_result, 'train', 'ExactGP', 'b')
    _plot_loss(axes[0, 1], pairwise_result, 'train', 'PairwiseGP', 'g')

    # ===== ROW 1: Validation Loss =====
    _plot_loss(axes[1, 0], exact_result, 'val', 'ExactGP', 'b')
    _plot_loss(axes[1, 1], pairwise_result, 'val', 'PairwiseGP', 'g')

    # ===== ROW 2: Function Fit =====
    if data.dimension == 1:
        _plot_function_fit_1d(
            axes[2, 0], exact_result, X_test_sorted, Y_test_sorted,
            X_train_flat, Y_train_flat, sort_idx, 'ExactGP', 'b',
            show_ground_truth=True
        )
        _plot_function_fit_1d(
            axes[2, 1], pairwise_result, X_test_sorted, Y_test_sorted,
            X_train_flat, Y_train_flat, sort_idx, 'PairwiseGP', 'g',
            show_ground_truth=False
        )
    else:
        _plot_nd_placeholder(axes[2, 0], 'ExactGP', data.dimension, exact_result is not None)
        _plot_nd_placeholder(axes[2, 1], 'PairwiseGP', data.dimension, pairwise_result is not None)

    # ===== ROW 3: Monotonicity (True vs Predicted) =====
    _plot_monotonicity(axes[3, 0], exact_result, Y_test, 'ExactGP', 'b')
    _plot_monotonicity(axes[3, 1], pairwise_result, Y_test, 'PairwiseGP', 'g')

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{fitness_fn}_4x2.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return pdf_path


def plot_from_saved_results(
    experiment_dir: Path,
    fitness_fn: str,
    data: ExperimentData,
) -> Optional[Path]:
    """
    Generate plot from saved experiment results.

    Loads ModelResult data from the models/ subdirectory.

    Args:
        experiment_dir: Experiment output directory.
        data: ExperimentData with X/Y arrays.
        fitness_fn: Fitness function name.

    Returns:
        Path to saved PDF, or None if no results.
    """
    models_dir = experiment_dir / "models"
    plots_dir = experiment_dir / "plots"

    # Load ExactGP result if available
    exact_result = _load_model_result(models_dir, "ExactGP", fitness_fn)
    pairwise_result = _load_model_result(models_dir, "PairwiseGP", fitness_fn)

    if exact_result is None and pairwise_result is None:
        return None

    return plot_fitness_function_grid(
        fitness_fn=fitness_fn,
        data=data,
        exact_result=exact_result,
        pairwise_result=pairwise_result,
        output_dir=plots_dir,
    )


def _load_model_result(models_dir: Path, gp_type: str, fitness_fn: str) -> Optional[ModelResult]:
    """Load a ModelResult from saved files."""
    model_dir = models_dir / f"{gp_type}_{fitness_fn}"

    if not model_dir.exists():
        return None

    # Load required files
    try:
        with open(model_dir / "hyperparams.json") as f:
            hyperparams = json.load(f)
        with open(model_dir / "metrics.json") as f:
            metrics = json.load(f)
        with open(model_dir / "losses.json") as f:
            losses = json.load(f)

        # Load predictions
        pred_file = model_dir / "predictions.csv"
        if pred_file.exists():
            pred_df = pd.read_csv(pred_file)
            # fold column is integer: 0=train, 1=val, 2=test
            y_pred_train = pred_df[pred_df['fold'] == 0]['y_pred'].values
            y_pred_val = pred_df[pred_df['fold'] == 1]['y_pred'].values
            y_pred_test = pred_df[pred_df['fold'] == 2]['y_pred'].values
            var_train = pred_df[pred_df['fold'] == 0]['variance'].values
            var_val = pred_df[pred_df['fold'] == 1]['variance'].values
            var_test = pred_df[pred_df['fold'] == 2]['variance'].values
        else:
            y_pred_train = y_pred_val = y_pred_test = None
            var_train = var_val = var_test = None

        return ModelResult(
            gp_type=gp_type,
            fitness_fn=fitness_fn,
            kernel_name=hyperparams.get('kernel', 'unknown'),
            seed=hyperparams.get('seed', 0),
            snr_data=hyperparams.get('snr_data', 0),
            snr_model=hyperparams.get('snr_model', 0),
            optimizer=hyperparams.get('optimizer', 'Adam'),
            lr=hyperparams.get('lr', 0),
            training_iters=hyperparams.get('training_iters', 0),
            signal_variance=metrics.get('signal_variance', 0),
            noise_variance_data=metrics.get('noise_variance_data', 0),
            noise_variance_model=hyperparams.get('noise_variance_model') or np.nan,
            lengthscale=hyperparams.get('lengthscale') or np.nan,
            train_mll=metrics.get('train_mll', 0),
            val_mll=metrics.get('val_mll', 0),
            test_mll=metrics.get('test_mll', 0),
            kendall_tau=metrics.get('kendall_tau', 0),
            spearman=metrics.get('spearman', 0),
            train_losses=losses.get('train_losses', []),
            val_losses=losses.get('val_losses', []),
            y_pred_train=y_pred_train,
            y_pred_val=y_pred_val,
            y_pred_test=y_pred_test,
            var_train=var_train,
            var_val=var_val,
            var_test=var_test,
        )
    except Exception as e:
        print(f"Warning: Could not load {gp_type}_{fitness_fn}: {e}")
        return None


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _plot_loss(ax, result: Optional[ModelResult], loss_type: str, model_name: str, color: str):
    """Plot training or validation loss curve."""
    losses = None
    if result is not None:
        losses = result.train_losses if loss_type == 'train' else result.val_losses

    if losses is not None and len(losses) > 0:
        ax.plot(range(1, len(losses) + 1), losses, f'{color}-', linewidth=1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (NLL)')
        loss_label = 'Training' if loss_type == 'train' else 'Validation'
        ax.set_title(f'{model_name}: {loss_label} Loss (LR={result.lr})')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'No {model_name} data',
                ha='center', va='center', transform=ax.transAxes)
        loss_label = 'Training' if loss_type == 'train' else 'Validation'
        ax.set_title(f'{model_name}: {loss_label} Loss')


def _plot_function_fit_1d(ax, result: Optional[ModelResult],
                          X_test_sorted, Y_test_sorted,
                          X_train_flat, Y_train_flat, sort_idx,
                          model_name: str, color: str, show_ground_truth: bool = False):
    """Plot 1D function fit with confidence intervals."""
    if result is not None and result.y_pred_test is not None:
        y_pred = result.y_pred_test.flatten()[sort_idx]
        var = result.var_test.flatten()[sort_idx] if result.var_test is not None else np.zeros_like(y_pred)
        std = np.sqrt(var)
        lower = y_pred - 2 * std
        upper = y_pred + 2 * std

        if show_ground_truth:
            ax.plot(X_test_sorted, Y_test_sorted, 'k--', label="Ground Truth", linewidth=1.5)

        ax.plot(X_test_sorted, y_pred, f'{color}-', label=f"{model_name} Mean", linewidth=1.5)
        ax.fill_between(X_test_sorted, lower, upper, color=color, alpha=0.2, label="95% CI")

        if show_ground_truth:
            ax.scatter(X_train_flat, Y_train_flat, c='k', marker='x', s=30,
                       label="Train Data", zorder=5)

        ax.set_xlabel('X')
        ylabel = 'Y' if model_name == 'ExactGP' else 'Latent Utility'
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name}: Function Fit (MLL={result.test_mll:.2f})")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'No {model_name} data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{model_name}: Function Fit")


def _plot_nd_placeholder(ax, model_name: str, dimension: int, has_data: bool):
    """Placeholder for N-D function fit (not supported for plotting)."""
    if has_data:
        ax.text(0.5, 0.5, f'{model_name}\n(D={dimension}, plotting not supported)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, f'No {model_name} data',
                ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f"{model_name}: Function Fit")


def _plot_monotonicity(ax, result: Optional[ModelResult], Y_test, model_name: str, color: str):
    """Plot true vs predicted (monotonicity check)."""
    if result is not None and result.y_pred_test is not None:
        y_pred = result.y_pred_test.flatten()
        var = result.var_test.flatten() if result.var_test is not None else np.zeros_like(y_pred)
        std = np.sqrt(var)
        y_true = Y_test.flatten()

        ax.errorbar(y_pred, y_true, xerr=2 * std, fmt='o', color=color, alpha=0.3, markersize=4)

        # Add ideal fit line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        if model_name == 'ExactGP':
            ax.plot(lims, lims, 'r--', alpha=0.75, label='Ideal Fit')

        xlabel = 'Predicted Y' if model_name == 'ExactGP' else 'Predicted Latent Utility'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('True Y')
        ax.set_title(f"{model_name}: Monotonicity\nTau={result.kendall_tau:.3f}, Spearman={result.spearman:.3f}")
        if model_name == 'ExactGP':
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'No {model_name} data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{model_name}: Monotonicity")
