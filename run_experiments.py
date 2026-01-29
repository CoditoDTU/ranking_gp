#!/usr/bin/env python
"""
Main experiment runner for GP ranking experiments.

This script orchestrates the experiment but does NOT define utility functions.
All logic is imported from src/ modules.

Usage:
    python run_experiments.py --config config.yaml
    python run_experiments.py --seed 42 --noise_type gaussian
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import warnings
import random
import torch
import numpy as np
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

from src.config import create_experiment_parser, create_experiment_config
from src.experiment import ExperimentManager, ResultsCollector, TrainingResult, ProgressTracker
from src.trainers import PairwiseGPTrainer, ExactGPTrainer
from src.fitness_functions import fitness_function
from src.datatools import get_comparisons
from src.noise import add_noise
from src.visualization import plot_experiment_grid


def main():
    warnings.filterwarnings("ignore", message="The input matches the stored training data")

    # --- Parse arguments and load config ---
    parser = create_experiment_parser()
    args = parser.parse_args()
    config = create_experiment_config(args.config, vars(args))

    # --- Set seeds ---
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup experiment directory and logging ---
    exp_manager = ExperimentManager(quiet=args.quiet)
    output_dir = exp_manager.setup()
    results = ResultsCollector(output_dir, exp_manager.experiment_id)

    # Save config at start for reproducibility
    results.save_config(config)

    # --- Setup progress tracking ---
    noise_types_list = ['none'] if not config.noise else config.noise_types
    total_experiments = (
        len(config.fitness_functions) *
        len(noise_types_list) *
        len(config.kernel_names) *
        2  # PairwiseGP + ExactGP
    )
    progress = ProgressTracker(total=total_experiments, quiet=args.quiet)
    progress.start()

    # --- Main experiment loop ---
    for fn_name in config.fitness_functions:
        progress.write(f"\n{'='*40}")
        progress.write(f"Fitness Function: {fn_name}")
        progress.write(f"{'='*40}")

        fitness_fn = fitness_function(base_fn_name=fn_name, dimension=config.dimension)

        # Generate all data per fitness function: sample N points and get true labels
        n_total = config.nsamples + config.n_test_points
        X_all = fitness_fn.sample_uniform(n_total, seed=config.seed)
        Y_all_true = torch.Tensor(fitness_fn.output(X_all))

        # Split into training and testing
        X_train = X_all[:config.nsamples]
        X_test = X_all[config.nsamples:]
        X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float64)

        Y_train = Y_all_true[:config.nsamples]
        Y_test_true = Y_all_true[config.nsamples:]

        noise_iterator = ['none'] if not config.noise else config.noise_types

        for noise_type in noise_iterator:

            if not config.noise:
                comparisons, X_train_pairwise = get_comparisons(Y_train, X=X_train_tensor)
                noise_level = 0.0
                noise_type = 'none'
                Y_noisy = None
                Y_test = Y_test_true
            else:
                # Noise all labels, then subset into train/test
                noise_level = config.noise_params.g_std
                Y_all_noisy = add_noise(
                    Y_all_true, noise_type=noise_type,
                    noise_params={'g_std': config.noise_params.g_std,
                                  'h_std': config.noise_params.h_std,
                                  'amplitude_factor': config.noise_params.amplitude_factor},
                )
                Y_noisy_train = Y_all_noisy[:config.nsamples]
                Y_noisy_test = Y_all_noisy[config.nsamples:]

                Y_noisy = Y_noisy_train
                Y_test = Y_noisy_test
                comparisons, X_train_pairwise = get_comparisons(Y_noisy_train, X=X_train_tensor)

            for kernel_name in config.kernel_names:
                # Build data dicts for results storage
                if not config.noise:
                    df_train_data = {
                        'fold': 0,
                        'X': X_train.flatten() if config.dimension == 1 else X_train.tolist(),
                        'y_true': Y_train.flatten().numpy(),
                    }
                    df_test_data = {
                        'fold': 1,
                        'X': X_test.flatten() if config.dimension == 1 else X_test.tolist(),
                        'y_true': Y_test_true.flatten().numpy(),
                    }
                else:
                    df_train_data = {
                        'fold': 0,
                        'X': X_train.flatten() if config.dimension == 1 else X_train.tolist(),
                        'y_true': Y_train.flatten().numpy(),
                        'y_noisy': Y_noisy_train.flatten().numpy(),
                    }
                    df_test_data = {
                        'fold': 1,
                        'X': X_test.flatten() if config.dimension == 1 else X_test.tolist(),
                        'y_true': Y_test_true.flatten().numpy(),
                        'y_noisy': Y_noisy_test.flatten().numpy(),
                    }

                # ========== PairwiseGP ==========
                progress.set_status(f"{fn_name}/{kernel_name}/PairwiseGP")
                _train_and_record(
                    PairwiseGPTrainer, config.pairwise_gp, kernel_name, config.dimension,
                    device, X_train_tensor, Y_train, X_test_tensor, Y_test,
                    comparisons, X_train_pairwise, X_train, X_test,
                    fn_name, noise_type, noise_level, config.seed,
                    results, df_train_data, df_test_data,
                )
                progress.update()

                # ========== ExactGP ==========
                progress.set_status(f"{fn_name}/{kernel_name}/ExactGP")
                Y_train_exact = Y_noisy if config.noise else Y_train
                _train_and_record(
                    ExactGPTrainer, config.exact_gp, kernel_name, config.dimension,
                    device, X_train_tensor, Y_train_exact, X_test_tensor, Y_test,
                    comparisons, X_train_pairwise, X_train, X_test,
                    fn_name, noise_type, noise_level, config.seed,
                    results, df_train_data, df_test_data,
                )
                progress.update()

    progress.finish()

    # --- Save results ---
    print("\n--- Saving Merged Predictions ---")
    results.save_predictions()
    results.save_losses()
    results.save_summary()
    results.save_aggregate(exp_manager.base_dir, config.clear_aggregate)

    # --- Generate plots ---
    if not args.no_plot:
        print("\n--- Generating Comprehensive Plots ---")
        for fn_name in results.prediction_data:
            for kernel_name in results.prediction_data[fn_name]:
                pdf_path = plot_experiment_grid(
                    fn_name, kernel_name,
                    results.prediction_data[fn_name][kernel_name],
                    results.training_losses.get(fn_name, {}).get(kernel_name, {}),
                    config.dimension,
                    config.exact_gp.lr,
                    config.pairwise_gp.lr,
                    exp_manager.plots_dir,
                )
                if pdf_path:
                    print(f"  Saved: {pdf_path}")
    else:
        print("\n--- Skipping plot generation (--no-plot) ---")

    print(f"\n--- Experiment Complete (elapsed: {progress.elapsed_str}) ---")

    # Print completion message to terminal (even in quiet mode)
    if args.quiet:
        import sys
        sys.stdout = sys.stdout.terminal  # Restore terminal output
        print(f"Experiment {exp_manager.experiment_id} complete in {progress.elapsed_str}. Results in: {output_dir}")


def _train_and_record(
    trainer_class, gp_settings, kernel_name, dimension, device,
    X_train_tensor, Y_train, X_test_tensor, Y_test,
    comparisons, X_train_pairwise, X_train_np, X_test_np,
    fn_name, noise_type, noise_level, seed,
    results, df_train_data, df_test_data,
):
    """Train a GP model and record results."""
    trainer = trainer_class(
        kernel_name=kernel_name,
        dimension=dimension,
        training_iters=gp_settings.training_iters,
        lr=gp_settings.lr,
        optimizer_name=gp_settings.optimizer,
        device=device,
    )

    try:
        print(f"Training {trainer.model_name}...")

        # Train
        losses = trainer.train(X_train_tensor, Y_train, comparisons, X_train_pairwise)

        # Predict
        y_pred_train, var_train = trainer.predict(X_train_tensor)
        y_pred_test, var_test = trainer.predict(X_test_tensor)

        # Train NLL is the last training loss
        train_nll = losses[-1]

        # Compute test NLL
        test_nll = trainer.compute_nll(X_test_tensor, Y_test, comparisons)

        # Compute ranking metrics
        tau, _ = kendalltau(Y_test.numpy().flatten(), y_pred_test.flatten())
        spearman, _ = spearmanr(Y_test.numpy().flatten(), y_pred_test.flatten())

        result = TrainingResult(
            model_name=trainer.model_name,
            fn_name=fn_name,
            kernel_name=kernel_name,
            noise_type=noise_type,
            dimension=dimension,
            seed=seed,
            noise_level=noise_level,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            variance_train=var_train,
            variance_test=var_test,
            train_nll=train_nll,
            test_nll=test_nll,
            kendall_tau=tau,
            spearman=spearman,
            lengthscale=trainer.get_lengthscale(),
            losses=losses,
            X_train=X_train_np.copy(),
            Y_train=Y_train.numpy().copy(),
            X_test=X_test_np.copy(),
            Y_test=Y_test.numpy().copy(),
        )

        results.add_result(result, df_train_data, df_test_data)

    except Exception as e:
        print(f"ERROR training {trainer.model_name} with {kernel_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
