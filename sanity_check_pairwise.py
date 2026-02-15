#!/usr/bin/env python
"""
Sanity check script for PairwiseGP training.

Verifies that the PairwiseGP predictions correctly rank the training data.
For a well-trained PairwiseGP, the predicted latent utilities should correctly
order the training points according to their true values.

Usage:
    python sanity_check_pairwise.py --fitness_function dixon_price --seed 5 --snr 100 --kernel squared_exponential
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

from src.data import ExperimentData
from src.models import PairwiseGPModel
from src.trainers import PairwiseGPTrainer
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="PairwiseGP Training Sanity Check")
    parser.add_argument("--fitness_function", type=str, default="dixon_price")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--snr", type=float, default=100)
    parser.add_argument("--kernel", type=str, default="squared_exponential")
    parser.add_argument("--config", type=str, default="configs/config_new.yaml")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=400)
    args = parser.parse_args()

    # Load config for default values
    config = load_config(args.config)

    print("=" * 60)
    print("PAIRWISE GP TRAINING SANITY CHECK")
    print("=" * 60)
    print(f"Fitness function: {args.fitness_function}")
    print(f"Seed: {args.seed}")
    print(f"SNR: {args.snr}")
    print(f"Kernel: {args.kernel}")
    print(f"LR: {args.lr}, Iterations: {args.iters}")
    print("=" * 60)

    # --- Prepare Data ---
    print("\n[1] Preparing data...")
    data = ExperimentData(
        fitness_fn_name=args.fitness_function,
        dimension=config.data.dimension,
        n_train=config.data.n_train,
        n_test=config.data.n_test,
        val_fraction=config.data.val_fraction,
        snr=args.snr,
        seed=args.seed,
    )
    data.prepare()

    print(f"    Training samples: {data.n_train_actual}")
    print(f"    Validation samples: {data.n_val}")
    print(f"    Test samples: {len(data.X_test)}")
    print(f"    Training comparisons: {len(data.comparisons_train)}")
    print(f"    Validation comparisons: {len(data.comparisons_val)}")
    print(f"    Signal variance (standardized): {data.signal_variance:.4f}")
    print(f"    Noise variance (standardized): {data.noise_variance:.4f}")

    # --- Build and Train Model ---
    print("\n[2] Building and training PairwiseGP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PairwiseGPModel(
        kernel_name=args.kernel,
        dimension=config.data.dimension,
        snr_model=config.model.snr_model,
        signal_variance=data.signal_variance,
        device=device,
    )
    model.build(
        X_train=data.X_train,
        y_train=data.Y_train_noisy,
        comparisons=data.comparisons_train,
        X_pairwise=data.X_train_pairwise,
    )

    trainer = PairwiseGPTrainer(
        model=model,
        training_iters=args.iters,
        lr=args.lr,
        optimizer_name="Adam",
        early_stopping=False,  # Disable for sanity check
    )

    train_losses, val_losses = trainer.train(data)
    print(f"    Final training loss: {train_losses[-1]:.4f}")

    # --- Get Predictions on Training Data ---
    print("\n[3] Getting predictions on training data...")
    y_pred_train, var_train = trainer.predict(data.X_train)
    y_pred_train = y_pred_train.flatten()

    # True values (standardized)
    y_true_train = data.Y_train_true.numpy().flatten()
    y_noisy_train = data.Y_train_noisy.numpy().flatten()

    # --- Sanity Check: Ranking Correlation ---
    print("\n[4] SANITY CHECK: Ranking Correlation")
    print("-" * 60)

    # Compute ranking metrics
    tau_true, _ = kendalltau(y_true_train, y_pred_train)
    spearman_true, _ = spearmanr(y_true_train, y_pred_train)
    tau_noisy, _ = kendalltau(y_noisy_train, y_pred_train)
    spearman_noisy, _ = spearmanr(y_noisy_train, y_pred_train)

    print(f"    Ranking vs TRUE values:")
    print(f"      Kendall tau:  {tau_true:.4f}")
    print(f"      Spearman rho: {spearman_true:.4f}")
    print()
    print(f"    Ranking vs NOISY values:")
    print(f"      Kendall tau:  {tau_noisy:.4f}")
    print(f"      Spearman rho: {spearman_noisy:.4f}")

    # Check pairwise accuracy
    print("\n[5] Pairwise Comparison Accuracy (on training pairs):")
    correct_pairs = 0
    total_pairs = len(data.comparisons_train)

    for i, j in data.comparisons_train:
        # True comparison: was y_noisy[i] > y_noisy[j]? (this is how comparisons are defined)
        true_comparison = y_noisy_train[i] > y_noisy_train[j]
        pred_comparison = y_pred_train[i] > y_pred_train[j]
        if true_comparison == pred_comparison:
            correct_pairs += 1

    pairwise_accuracy = correct_pairs / total_pairs * 100
    print(f"    Correct pairs: {correct_pairs}/{total_pairs} ({pairwise_accuracy:.1f}%)")

    # --- Get predictions on test data ---
    print("\n[6] Test Set Performance:")
    y_pred_test, var_test = trainer.predict(data.X_test)
    y_pred_test = y_pred_test.flatten()
    y_true_test = data.Y_test_true.numpy().flatten()

    tau_test, _ = kendalltau(y_true_test, y_pred_test)
    spearman_test, _ = spearmanr(y_true_test, y_pred_test)
    print(f"    Kendall tau:  {tau_test:.4f}")
    print(f"    Spearman rho: {spearman_test:.4f}")

    # --- Check learned hyperparameters ---
    print("\n[7] Learned hyperparameters:")
    print(f"    Lengthscale: {trainer.get_lengthscale():.4f}")

    # --- Visualization ---
    print("\n[8] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"PairwiseGP Sanity Check: {args.fitness_function} (SNR={args.snr}, seed={args.seed})",
                 fontsize=14, fontweight='bold')

    # Plot 1: Training loss
    ax = axes[0, 0]
    ax.plot(train_losses, 'b-', label='Train')
    if val_losses:
        ax.plot(val_losses, 'r--', label='Val')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (NLL)')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Predicted utility vs True value (training data)
    ax = axes[0, 1]
    ax.scatter(y_true_train, y_pred_train, alpha=0.6, s=40)
    ax.set_xlabel('True Y (standardized)')
    ax.set_ylabel('Predicted Latent Utility')
    ax.set_title(f'Train: Utility vs True Y\nTau={tau_true:.3f}, Spearman={spearman_true:.3f}')
    ax.grid(True, alpha=0.3)
    # Add trend line
    z = np.polyfit(y_true_train, y_pred_train, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_true_train.min(), y_true_train.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Linear fit')
    ax.legend()

    # Plot 3: Predicted utility vs True value (test data)
    ax = axes[1, 0]
    ax.scatter(y_true_test, y_pred_test, alpha=0.6, s=40, c='green')
    ax.set_xlabel('True Y (standardized)')
    ax.set_ylabel('Predicted Latent Utility')
    ax.set_title(f'Test: Utility vs True Y\nTau={tau_test:.3f}, Spearman={spearman_test:.3f}')
    ax.grid(True, alpha=0.3)
    # Add trend line
    z = np.polyfit(y_true_test, y_pred_test, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_true_test.min(), y_true_test.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Linear fit')
    ax.legend()

    # Plot 4: Function fit (if 1D)
    ax = axes[1, 1]
    if config.data.dimension == 1:
        X_test = data.X_test.numpy().flatten()
        sort_idx = np.argsort(X_test)
        X_test_sorted = X_test[sort_idx]

        # Predictions (latent utilities, keep standardized scale)
        y_pred_sorted = y_pred_test[sort_idx]
        std_sorted = np.sqrt(var_test.flatten()[sort_idx])

        # True Y (standardized for comparison)
        Y_test_std = y_true_test[sort_idx]

        # Normalize both to same scale for visualization
        y_pred_norm = (y_pred_sorted - y_pred_sorted.mean()) / y_pred_sorted.std()
        Y_test_norm = (Y_test_std - Y_test_std.mean()) / Y_test_std.std()

        ax.plot(X_test_sorted, Y_test_norm, 'k--', label='True Y (normalized)', linewidth=1.5)
        ax.plot(X_test_sorted, y_pred_norm, 'g-', label='Latent Utility (normalized)', linewidth=1.5)

        # Training points
        X_train = data.X_train.numpy().flatten()
        y_train_norm = (y_noisy_train - y_noisy_train.mean()) / y_noisy_train.std()
        ax.scatter(X_train, y_train_norm, c='k', marker='x', s=30, label='Train Data', zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Function Shape Comparison')
        ax.legend(loc='best', fontsize=8)
    else:
        ax.text(0.5, 0.5, f'N-D plot not supported\n(dimension={config.data.dimension})',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Function Fit')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = f"sanity_check_pairwise_{args.fitness_function}_s{args.seed}_snr{args.snr}.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"    Saved plot to: {output_path}")
    plt.close()

    # --- Final Verdict ---
    print("\n" + "=" * 60)
    if tau_true >= 0.8 and pairwise_accuracy >= 90:
        print("SANITY CHECK PASSED")
        print("PairwiseGP correctly ranks training data with high accuracy.")
    elif tau_true >= 0.6 and pairwise_accuracy >= 75:
        print("SANITY CHECK WARNING")
        print("PairwiseGP ranking is reasonable but could be improved.")
    else:
        print("SANITY CHECK FAILED")
        print("PairwiseGP ranking is poor. Check hyperparameters or noise level.")
    print("=" * 60)

    trainer.cleanup()


if __name__ == "__main__":
    main()
