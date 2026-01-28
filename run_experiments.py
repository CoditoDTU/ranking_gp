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

import warnings
import random
import torch
import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.config import create_experiment_parser, create_experiment_config
from src.experiment import ExperimentManager, ResultsCollector, TrainingResult
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
    exp_manager = ExperimentManager()
    output_dir = exp_manager.setup()
    results = ResultsCollector(output_dir, exp_manager.experiment_id)

    # --- Main experiment loop ---
    for fn_name in config.fitness_functions:
        print(f"\n======================================")
        print(f"Testing Fitness Function: {fn_name}")
        print(f"======================================")

        fitness_fn = fitness_function(base_fn_name=fn_name, dimension=config.dimension)

        # Generate common test data
        total_test_points = 100
        grid_points_per_dim = int(total_test_points ** (1 / config.dimension))
        X_test = fitness_fn.sample_grids(grid_points_per_dim)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
        Y_test = torch.Tensor(fitness_fn.output(X_test))

        noise_iterator = ['none'] if not config.noise else config.noise_types

        for noise_type in noise_iterator:

            # Generate training data
            X_train = fitness_fn.sample_uniform(config.nsamples, seed=config.seed)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
            Y_train = torch.Tensor(fitness_fn.output(X_train))

            if not config.noise:
                comparisons, X_train_pairwise = get_comparisons(Y_train, X=X_train_tensor)
                print(f"\n--- No Noise ---")
                noise_level = 0.0
                noise_type = 'none'
                Y_noisy = None
            else:
                noise_level = config.noise_params.g_std
                print(f"\n--- Noise Type: {noise_type} ---")
                Y_noisy = add_noise(
                    Y_train, noise_type='gaussian',
                    noise_params={'g_std': config.noise_params.g_std,
                                  'h_std': config.noise_params.h_std,
                                  'amplitude_factor': config.noise_params.amplitude_factor},
                )
                comparisons, X_train_pairwise = get_comparisons(Y_noisy, X=X_train_tensor)

            for kernel_name in config.kernel_names:
                print(f"\n----- Kernel: {kernel_name} -----")

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
                        'y_true': Y_test.flatten().numpy(),
                    }
                else:
                    df_train_data = {
                        'fold': 0,
                        'X': X_train.flatten() if config.dimension == 1 else X_train.tolist(),
                        'y_true': Y_train.flatten().numpy(),
                        'y_noisy': Y_noisy.flatten().numpy(),
                    }
                    df_test_data = {
                        'fold': 1,
                        'X': X_test.flatten() if config.dimension == 1 else X_test.tolist(),
                        'y_true': Y_test.flatten().numpy(),
                        'y_noisy': np.nan,
                    }

                # ========== PairwiseGP ==========
                _train_and_record(
                    PairwiseGPTrainer, config.pairwise_gp, kernel_name, config.dimension,
                    device, X_train_tensor, Y_train, X_test_tensor, Y_test,
                    comparisons, X_train_pairwise, X_train, X_test,
                    fn_name, noise_type, noise_level, config.seed,
                    results, df_train_data, df_test_data,
                )

                # ========== ExactGP ==========
                Y_train_exact = Y_noisy if config.noise else Y_train
                _train_and_record(
                    ExactGPTrainer, config.exact_gp, kernel_name, config.dimension,
                    device, X_train_tensor, Y_train_exact, X_test_tensor, Y_test,
                    comparisons, X_train_pairwise, X_train, X_test,
                    fn_name, noise_type, noise_level, config.seed,
                    results, df_train_data, df_test_data,
                )

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

    print("\n--- Experiment Complete ---")


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
