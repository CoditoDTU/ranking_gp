#!/usr/bin/env python
"""
Sanity check script for GP training.

Verifies that the GP predictions on training data are within the noise bounds.
For a well-trained GP, the mean predictions on training points should be close
to the noisy observations (within ± noise_std).

Usage:
    python sanity_check.py --fitness_function dixon_price --seed 5 --snr 100 --kernel exponential
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data import ExperimentData
from src.models import ExactGPModel
from src.trainers import ExactGPTrainer
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="GP Training Sanity Check")
    parser.add_argument("--fitness_function", type=str, default="dixon_price")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--snr", type=float, default=100)
    parser.add_argument("--kernel", type=str, default="exponential")
    parser.add_argument("--config", type=str, default="configs/config_new.yaml")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=400)
    args = parser.parse_args()

    # Load config for default values
    config = load_config(args.config)

    print("=" * 60)
    print("GP TRAINING SANITY CHECK")
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
    print(f"    y_mean (original scale): {data.y_mean:.4f}")
    print(f"    y_std (original scale): {data.y_std:.4f}")
    print(f"    Signal variance (standardized): {data.signal_variance:.4f}")
    print(f"    Noise variance (standardized): {data.noise_variance:.4f}")
    print(f"    Noise std (standardized): {np.sqrt(data.noise_variance):.4f}")

    # --- Build and Train Model ---
    print("\n[2] Building and training ExactGP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ExactGPModel(
        kernel_name=args.kernel,
        dimension=config.data.dimension,
        snr_model=config.model.snr_model,
        signal_variance=data.signal_variance,
        device=device,
    )
    model.build(X_train=data.X_train, y_train=data.Y_train_noisy)

    trainer = ExactGPTrainer(
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
    std_train = np.sqrt(var_train)

    # Get the noisy training labels (standardized)
    y_noisy_train = data.Y_train_noisy.numpy().flatten()

    # --- Sanity Check: Are predictions within noise bounds? ---
    print("\n[4] SANITY CHECK: Training data predictions vs noisy observations")
    print("-" * 60)

    noise_std = np.sqrt(data.noise_variance)
    residuals = y_pred_train.flatten() - y_noisy_train

    # Check how many predictions are within 1, 2, 3 noise stds of observations
    within_1_std = np.sum(np.abs(residuals) <= noise_std) / len(residuals) * 100
    within_2_std = np.sum(np.abs(residuals) <= 2 * noise_std) / len(residuals) * 100
    within_3_std = np.sum(np.abs(residuals) <= 3 * noise_std) / len(residuals) * 100

    print(f"    Noise std (standardized): {noise_std:.4f}")
    print(f"    Mean absolute residual: {np.mean(np.abs(residuals)):.4f}")
    print(f"    Max absolute residual: {np.max(np.abs(residuals)):.4f}")
    print()
    print(f"    Predictions within ±1 noise_std: {within_1_std:.1f}%")
    print(f"    Predictions within ±2 noise_std: {within_2_std:.1f}%")
    print(f"    Predictions within ±3 noise_std: {within_3_std:.1f}%")

    # Expected percentages for Gaussian: 68.3%, 95.4%, 99.7%
    print()
    print("    Expected (if residuals ~ N(0, noise_std²)):")
    print("    ±1 std: 68.3%, ±2 std: 95.4%, ±3 std: 99.7%")

    # --- Check learned hyperparameters ---
    print("\n[5] Learned hyperparameters:")
    print(f"    Lengthscale: {trainer.get_lengthscale():.4f}")
    print(f"    Noise variance (model): {trainer.get_noise_variance():.4f}")
    print(f"    Noise variance (data): {data.noise_variance:.4f}")

    # --- Visualization ---
    print("\n[6] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Sanity Check: {args.fitness_function} (SNR={args.snr}, seed={args.seed})",
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

    # Plot 2: Predictions vs Noisy Observations (training data)
    ax = axes[0, 1]
    ax.errorbar(y_noisy_train, y_pred_train.flatten(), yerr=2*std_train.flatten(),
                fmt='o', alpha=0.5, markersize=4, label='Pred ± 2σ')
    lims = [min(y_noisy_train.min(), y_pred_train.min()) - 0.5,
            max(y_noisy_train.max(), y_pred_train.max()) + 0.5]
    ax.plot(lims, lims, 'r--', label='Ideal')
    ax.set_xlabel('Noisy Observation (standardized)')
    ax.set_ylabel('GP Prediction (standardized)')
    ax.set_title('Training Data: Pred vs Obs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=20, density=True, alpha=0.7, label='Residuals')
    # Overlay expected Gaussian
    x_gauss = np.linspace(-4*noise_std, 4*noise_std, 100)
    y_gauss = (1 / (noise_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_gauss / noise_std) ** 2)
    ax.plot(x_gauss, y_gauss, 'r-', linewidth=2, label=f'N(0, {noise_std:.3f}²)')
    ax.axvline(-noise_std, color='g', linestyle='--', alpha=0.7)
    ax.axvline(noise_std, color='g', linestyle='--', alpha=0.7, label='±1 noise_std')
    ax.set_xlabel('Residual (pred - obs)')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Function fit (if 1D)
    ax = axes[1, 1]
    if config.data.dimension == 1:
        X_test = data.X_test.numpy().flatten()
        sort_idx = np.argsort(X_test)
        X_test_sorted = X_test[sort_idx]

        # De-standardize for plotting
        Y_test_true = data.destandardize_y(data.Y_test_true.numpy().flatten())[sort_idx]

        y_pred_test, var_test = trainer.predict(data.X_test)
        y_pred_test_orig, var_test_orig = data.destandardize_predictions(y_pred_test, var_test)
        y_pred_sorted = y_pred_test_orig.flatten()[sort_idx]
        std_sorted = np.sqrt(var_test_orig.flatten()[sort_idx])

        ax.plot(X_test_sorted, Y_test_true, 'k--', label='Ground Truth', linewidth=1.5)
        ax.plot(X_test_sorted, y_pred_sorted, 'b-', label='GP Mean', linewidth=1.5)
        ax.fill_between(X_test_sorted,
                        y_pred_sorted - 2*std_sorted,
                        y_pred_sorted + 2*std_sorted,
                        alpha=0.2, color='b', label='95% CI')

        # Training points (de-standardized)
        X_train = data.X_train.numpy().flatten()
        Y_train_noisy_orig = data.destandardize_y(data.Y_train_noisy.numpy().flatten())
        ax.scatter(X_train, Y_train_noisy_orig, c='k', marker='x', s=30, label='Train Data', zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y (original scale)')
        ax.set_title('Function Fit (Test Data)')
        ax.legend(loc='best', fontsize=8)
    else:
        ax.text(0.5, 0.5, f'N-D plot not supported\n(dimension={config.data.dimension})',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Function Fit')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = f"sanity_check_{args.fitness_function}_s{args.seed}_snr{args.snr}.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"    Saved plot to: {output_path}")
    plt.close()

    # --- Final Verdict ---
    print("\n" + "=" * 60)
    if within_2_std >= 90:
        print("SANITY CHECK PASSED")
        print("GP predictions on training data are within expected noise bounds.")
    elif within_2_std >= 70:
        print("SANITY CHECK WARNING")
        print("GP predictions are somewhat close to training data, but may need tuning.")
    else:
        print("SANITY CHECK FAILED")
        print("GP predictions deviate significantly from training observations.")
        print("This may indicate underfitting, wrong hyperparameters, or a bug.")
    print("=" * 60)

    trainer.cleanup()


if __name__ == "__main__":
    main()
