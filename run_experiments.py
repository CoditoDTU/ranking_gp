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
from src.experiment import ExperimentManager, ResultsCollector, TrainingResult, ProgressTracker
from src.trainers import PairwiseGPTrainer, ExactGPTrainer
from src.fitness_functions import fitness_function
from src.datatools import get_comparisons
from src.noise import add_noise


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

        Y_train_all = Y_all_true[:config.nsamples]
        Y_test_true = Y_all_true[config.nsamples:]

        # --- Validation split ---
        has_val = config.val_fraction > 0.0
        if has_val:
            n_val = int(config.nsamples * config.val_fraction)
            n_train_reduced = config.nsamples - n_val
            perm = np.random.permutation(config.nsamples)
            train_idx = perm[:n_train_reduced]
            val_idx = perm[n_train_reduced:]

            X_train_reduced = X_train[train_idx]
            X_val = X_train[val_idx]
            X_train_reduced_tensor = torch.tensor(X_train_reduced, dtype=torch.float64)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float64)

            Y_train = Y_train_all[train_idx]
            Y_val_true = Y_train_all[val_idx]
        else:
            X_train_reduced = X_train
            X_train_reduced_tensor = X_train_tensor
            X_val = None
            X_val_tensor = None
            Y_train = Y_train_all
            Y_val_true = None

        noise_iterator = ['none'] if not config.noise else config.noise_types

        for noise_type in noise_iterator:

            if not config.noise:
                comparisons, X_train_pairwise = get_comparisons(Y_train, X=X_train_reduced_tensor)
                noise_level = 0.0
                noise_type = 'none'
                Y_noisy = None
                Y_test = Y_test_true
                Y_val = Y_val_true if has_val else None

                if has_val:
                    val_comparisons, X_val_pairwise = get_comparisons(Y_val_true, X=X_val_tensor)
                else:
                    val_comparisons, X_val_pairwise = None, None
            else:
                # Noise all labels, then subset into train/test/val
                noise_level = config.noise_params.g_std
                Y_all_noisy = add_noise(
                    Y_all_true, noise_type=noise_type,
                    noise_params={'g_std': config.noise_params.g_std,
                                  'h_std': config.noise_params.h_std,
                                  'amplitude_factor': config.noise_params.amplitude_factor},
                )
                Y_noisy_train_all = Y_all_noisy[:config.nsamples]
                Y_noisy_test = Y_all_noisy[config.nsamples:]

                if has_val:
                    Y_noisy_train = Y_noisy_train_all[train_idx]
                    Y_noisy_val = Y_noisy_train_all[val_idx]
                else:
                    Y_noisy_train = Y_noisy_train_all
                    Y_noisy_val = None

                Y_noisy = Y_noisy_train
                Y_test = Y_noisy_test
                Y_val = Y_noisy_val if has_val else None
                comparisons, X_train_pairwise = get_comparisons(Y_noisy_train, X=X_train_reduced_tensor)

                if has_val:
                    val_comparisons, X_val_pairwise = get_comparisons(Y_noisy_val, X=X_val_tensor)
                else:
                    val_comparisons, X_val_pairwise = None, None

            for kernel_name in config.kernel_names:
                # Build data dicts for results storage
                if not config.noise:
                    df_train_data = {
                        'fold': 0,
                        'X': X_train_reduced.flatten() if config.dimension == 1 else X_train_reduced.tolist(),
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
                        'X': X_train_reduced.flatten() if config.dimension == 1 else X_train_reduced.tolist(),
                        'y_true': Y_train.flatten().numpy(),
                        'y_noisy': Y_noisy_train.flatten().numpy(),
                    }
                    df_test_data = {
                        'fold': 1,
                        'X': X_test.flatten() if config.dimension == 1 else X_test.tolist(),
                        'y_true': Y_test_true.flatten().numpy(),
                        'y_noisy': Y_noisy_test.flatten().numpy(),
                    }

                # Build validation data dict
                df_val_data = None
                if has_val:
                    if not config.noise:
                        df_val_data = {
                            'fold': 2,
                            'X': X_val.flatten() if config.dimension == 1 else X_val.tolist(),
                            'y_true': Y_val_true.flatten().numpy(),
                        }
                    else:
                        df_val_data = {
                            'fold': 2,
                            'X': X_val.flatten() if config.dimension == 1 else X_val.tolist(),
                            'y_true': Y_val_true.flatten().numpy(),
                            'y_noisy': Y_noisy_val.flatten().numpy(),
                        }

                # ========== PairwiseGP ==========
                progress.set_status(f"{fn_name}/{kernel_name}/PairwiseGP")
                _train_and_record(
                    PairwiseGPTrainer, config.pairwise_gp, kernel_name, config.dimension,
                    device, X_train_reduced_tensor, Y_train, X_test_tensor, Y_test,
                    comparisons, X_train_pairwise, X_train_reduced, X_test,
                    fn_name, noise_type, noise_level, config.seed,
                    results, df_train_data, df_test_data,
                    X_val_tensor=X_val_tensor, Y_val=Y_val,
                    val_comparisons=val_comparisons, X_val_pairwise=X_val_pairwise,
                    X_val_np=X_val, df_val_data=df_val_data,
                )
                progress.update()

                # ========== ExactGP ==========
                progress.set_status(f"{fn_name}/{kernel_name}/ExactGP")
                Y_train_exact = Y_noisy if config.noise else Y_train
                _train_and_record(
                    ExactGPTrainer, config.exact_gp, kernel_name, config.dimension,
                    device, X_train_reduced_tensor, Y_train_exact, X_test_tensor, Y_test,
                    comparisons, X_train_pairwise, X_train_reduced, X_test,
                    fn_name, noise_type, noise_level, config.seed,
                    results, df_train_data, df_test_data,
                    X_val_tensor=X_val_tensor, Y_val=Y_val,
                    val_comparisons=val_comparisons, X_val_pairwise=X_val_pairwise,
                    X_val_np=X_val, df_val_data=df_val_data,
                )
                progress.update()

    progress.finish()

    # --- Save results ---
    print("\n--- Saving Merged Predictions ---")
    results.save_predictions()
    results.save_losses()
    if config.val_fraction > 0.0:
        results.save_val_losses()
    results.save_summary()
    results.save_aggregate(exp_manager.base_dir, config.clear_aggregate)

    print(f"\n--- Experiment Complete (elapsed: {progress.elapsed_str}) ---")
    print("To generate plots, run: python run_visualization.py")

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
    X_val_tensor=None, Y_val=None, val_comparisons=None, X_val_pairwise=None,
    X_val_np=None, df_val_data=None,
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
        losses, val_losses = trainer.train(
            X_train_tensor, Y_train, comparisons, X_train_pairwise,
            X_val=X_val_tensor, y_val=Y_val,
            val_comparisons=val_comparisons, X_val_pairwise=X_val_pairwise,
        )

        # Predict
        y_pred_train, var_train = trainer.predict(X_train_tensor)
        y_pred_test, var_test = trainer.predict(X_test_tensor)

        # Predict on validation set
        if X_val_tensor is not None:
            y_pred_val, var_val = trainer.predict(X_val_tensor)
            val_nll = trainer.compute_nll(X_val_tensor, Y_val, val_comparisons)
        else:
            y_pred_val, var_val, val_nll = None, None, None

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
            y_pred_val=y_pred_val,
            variance_val=var_val,
            val_nll=val_nll,
            val_losses=val_losses,
            X_val=X_val_np.copy() if X_val_np is not None else None,
            Y_val=Y_val.numpy().copy() if Y_val is not None else None,
        )

        results.add_result(result, df_train_data, df_test_data, df_val_data)

    except Exception as e:
        print(f"ERROR training {trainer.model_name} with {kernel_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
