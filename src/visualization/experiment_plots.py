"""
Comprehensive experiment plots (3x2 grid).
Extracted from module_1.py lines 630-798.

Rows: Training Loss | Function Fit | Monotonicity (True vs Predicted)
Columns: ExactGP | PairwiseGP
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_experiment_grid(
    fn_name: str,
    kernel_name: str,
    prediction_data: Dict[str, Dict],
    training_losses: Dict[str, List[float]],
    dimension: int,
    exact_lr: float,
    pairwise_lr: float,
    output_dir: str,
) -> Optional[str]:
    """
    Generate a 3x2 grid plot for a function/kernel combination.

    Args:
        fn_name: Fitness function name.
        kernel_name: Kernel name.
        prediction_data: Dict keyed by model name ('ExactGP', 'PairwiseGP')
                         with prediction dicts containing X_train, Y_train,
                         X_test, Y_test, y_pred, std, tau, spearman, test_nll.
        training_losses: Dict keyed by model name with loss lists.
        dimension: Input dimension.
        exact_lr: ExactGP learning rate (for title).
        pairwise_lr: PairwiseGP learning rate (for title).
        output_dir: Directory to save the plot PDF.

    Returns:
        Path to saved PDF, or None if no data.
    """
    has_exact = 'ExactGP' in prediction_data
    has_pairwise = 'PairwiseGP' in prediction_data

    if not (has_exact or has_pairwise):
        return None

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'{fn_name} - {kernel_name}', fontsize=14, fontweight='bold')

    # Get common data from whichever GP is available
    ref_gp = 'ExactGP' if has_exact else 'PairwiseGP'
    X_train = prediction_data[ref_gp]['X_train']
    Y_train = prediction_data[ref_gp]['Y_train']
    X_test = prediction_data[ref_gp]['X_test']
    Y_test = prediction_data[ref_gp]['Y_test']

    # For 1D, flatten and sort for proper line plots
    if dimension == 1:
        X_test_flat = X_test.flatten()
        sort_idx = np.argsort(X_test_flat)
        X_test_sorted = X_test_flat[sort_idx]
        Y_test_sorted = Y_test.flatten()[sort_idx]
        X_train_flat = X_train.flatten()
        Y_train_flat = Y_train.flatten()
    else:
        sort_idx = None
        X_test_sorted = Y_test_sorted = X_train_flat = Y_train_flat = None

    # ===== ROW 0: Training Loss =====
    _plot_training_loss(axes[0, 0], training_losses.get('ExactGP'), 'ExactGP', exact_lr, 'b')
    _plot_training_loss(axes[0, 1], training_losses.get('PairwiseGP'), 'PairwiseGP', pairwise_lr, 'g')

    # ===== ROW 1: Function Fit =====
    if dimension == 1:
        _plot_function_fit_1d(
            axes[1, 0], prediction_data.get('ExactGP'),
            X_test_sorted, Y_test_sorted, X_train_flat, Y_train_flat,
            sort_idx, 'ExactGP', 'b', show_ground_truth=True,
        )
        _plot_function_fit_1d(
            axes[1, 1], prediction_data.get('PairwiseGP'),
            X_test_sorted, Y_test_sorted, X_train_flat, Y_train_flat,
            sort_idx, 'PairwiseGP', 'g', show_ground_truth=False,
        )
    else:
        _plot_nd_placeholder(axes[1, 0], 'ExactGP', dimension, has_exact)
        _plot_nd_placeholder(axes[1, 1], 'PairwiseGP', dimension, has_pairwise)

    # ===== ROW 2: Monotonicity (True vs Predicted) =====
    _plot_monotonicity(axes[2, 0], prediction_data.get('ExactGP'), Y_test, 'ExactGP', 'b')
    _plot_monotonicity(axes[2, 1], prediction_data.get('PairwiseGP'), Y_test, 'PairwiseGP', 'g')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, f"{fn_name}_{kernel_name}.pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return pdf_path


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _plot_training_loss(ax, losses, model_name, lr, color):
    """Plot training loss curve."""
    if losses:
        ax.plot(range(1, len(losses) + 1), losses, f'{color}-', linewidth=1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (NLL)')
        ax.set_title(f'{model_name}: Training Loss (LR={lr})')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'No {model_name} loss data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name}: Training Loss')


def _plot_function_fit_1d(ax, data, X_test_sorted, Y_test_sorted,
                          X_train_flat, Y_train_flat, sort_idx,
                          model_name, color, show_ground_truth=False):
    """Plot 1D function fit with confidence intervals."""
    if data is not None:
        y_pred = data['y_pred'].flatten()[sort_idx]
        std = data['std'].flatten()[sort_idx]
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
        ax.set_title(f"{model_name}: Function Fit (NLL={data['test_nll']:.2f})")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'No {model_name} data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{model_name}: Function Fit")


def _plot_nd_placeholder(ax, model_name, dimension, has_data):
    """Placeholder for N-D function fit (not supported for plotting)."""
    if has_data:
        ax.text(0.5, 0.5, f'{model_name}\n(D={dimension}, plotting not supported)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, f'No {model_name} data',
                ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f"{model_name}: Function Fit")


def _plot_monotonicity(ax, data, Y_test, model_name, color):
    """Plot true vs predicted (monotonicity check)."""
    if data is not None:
        y_pred = data['y_pred'].flatten()
        std = data['std'].flatten()
        y_true = Y_test.flatten()

        ax.errorbar(y_pred, y_true, xerr=2 * std, fmt='o', color=color, alpha=0.3, markersize=4)

        # Add ideal fit line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.75, label='Ideal Fit') if model_name == 'ExactGP' else None

        xlabel = 'Predicted Y' if model_name == 'ExactGP' else 'Predicted Latent Utility'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('True Y')
        ax.set_title(f"{model_name}: Monotonicity\nTau={data['tau']:.3f}, Spearman={data['spearman']:.3f}")
        if model_name == 'ExactGP':
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'No {model_name} data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{model_name}: Monotonicity")
