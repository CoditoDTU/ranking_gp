#!/usr/bin/env python
"""
Main experiment runner for GP ranking experiments.

Uses the refactored module structure:
- src/data: ExperimentData for data handling
- src/models: ExactGPModel, PairwiseGPModel wrappers
- src/trainers: ExactGPTrainer, PairwiseGPTrainer
- src/results: ResultsCollector, ModelResult
- src/config: Config dataclasses and CLI parser

Usage:
    python run_experiments.py --config config_new.yaml
    python run_experiments.py --config config_new.yaml --seed 42 --snr 10.0
    python run_experiments.py --seed 42 --snr 20 --optimizer Adam --quiet
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import kendalltau, spearmanr

from src.config import create_experiment_parser, load_config_with_overrides
from src.data import ExperimentData
from src.models import ExactGPModel, PairwiseGPModel
from src.trainers import ExactGPTrainer, PairwiseGPTrainer
from src.results import ResultsCollector, ModelResult, FailureRecord


def main():
    warnings.filterwarnings("ignore", message="The input matches the stored training data")

    # --- Parse arguments and load config ---
    parser = create_experiment_parser()
    args = parser.parse_args()

    # Load config with CLI overrides
    config = load_config_with_overrides(args.config, vars(args))

    # --- Set seeds ---
    np.random.seed(config.experiment.seed)
    torch.manual_seed(config.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        print(f"Using device: {device}")
        print(f"Seed: {config.experiment.seed}")
        print(f"Data SNR: {config.data.snr}")
        print(f"Model SNR: {config.model.snr_model}")

    # --- Setup output directory ---
    # If output_dir was explicitly set via CLI, use it directly (for grid search)
    # Otherwise, create a timestamped subfolder
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.experiment.output_dir) / f"exp_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup results collector ---
    collector = ResultsCollector(output_dir, criterion=config.experiment.selection_criterion)

    # --- Count total experiments ---
    total = (
        len(config.data.fitness_functions) *
        len(config.model.kernels) *
        2  # ExactGP + PairwiseGP
    )
    current = 0

    # --- Main experiment loop ---
    for fn_name in config.data.fitness_functions:
        if not args.quiet:
            print(f"\n{'='*50}")
            print(f"Fitness Function: {fn_name}")
            print(f"{'='*50}")

        # Prepare data using ExperimentData
        data = ExperimentData(
            fitness_fn_name=fn_name,
            dimension=config.data.dimension,
            n_train=config.data.n_train,
            n_test=config.data.n_test,
            val_fraction=config.data.val_fraction,
            snr=config.data.snr,
            seed=config.experiment.seed,
        )
        data.prepare()

        for kernel_name in config.model.kernels:
            # ========== ExactGP ==========
            current += 1
            if not args.quiet:
                print(f"\n[{current}/{total}] ExactGP - {kernel_name}")

            result = train_exactgp(
                data=data,
                kernel_name=kernel_name,
                config=config,
                device=device,
                fn_name=fn_name,
                collector=collector,
            )
            if result is not None:
                collector.add_result(result)

            # ========== PairwiseGP ==========
            current += 1
            if not args.quiet:
                print(f"\n[{current}/{total}] PairwiseGP - {kernel_name}")

            result = train_pairwisegp(
                data=data,
                kernel_name=kernel_name,
                config=config,
                device=device,
                fn_name=fn_name,
                collector=collector,
            )
            if result is not None:
                collector.add_result(result)

    # --- Save results ---
    if not args.quiet:
        print(f"\n{'='*50}")
        print("Saving results...")

    collector.save()

    print(f"\nExperiment complete. Results saved to: {output_dir}")


def train_exactgp(data, kernel_name, config, device, fn_name, collector):
    """Train ExactGP model and return ModelResult."""
    try:
        # Build model
        model = ExactGPModel(
            kernel_name=kernel_name,
            dimension=config.data.dimension,
            snr_model=config.model.snr_model,
            signal_variance=data.signal_variance,
            device=device,
        )
        model.build(
            X_train=data.X_train,
            y_train=data.Y_train_noisy,
        )

        # Create trainer
        trainer = ExactGPTrainer(
            model=model,
            training_iters=config.trainer.exact_gp.training_iters,
            lr=config.trainer.exact_gp.lr,
            optimizer_name=config.trainer.exact_gp.optimizer,
        )

        # Train
        train_losses, val_losses = trainer.train(data)

        # Compute metrics
        train_mll = trainer.compute_mll(data.X_train, data.Y_train_noisy)
        val_mll = trainer.compute_mll(data.X_val, data.Y_val_noisy)
        test_mll = trainer.compute_mll(data.X_test, data.Y_test_noisy)

        # Predictions
        y_pred_train, var_train = trainer.predict(data.X_train)
        y_pred_val, var_val = trainer.predict(data.X_val)
        y_pred_test, var_test = trainer.predict(data.X_test)

        # Ranking metrics (on test set with true values)
        tau, _ = kendalltau(data.Y_test_true.numpy().flatten(), y_pred_test.flatten())
        spearman, _ = spearmanr(data.Y_test_true.numpy().flatten(), y_pred_test.flatten())

        result = ModelResult(
            gp_type="ExactGP",
            fitness_fn=fn_name,
            kernel_name=kernel_name,
            seed=config.experiment.seed,
            snr_data=config.data.snr,
            snr_model=config.model.snr_model,
            optimizer=config.trainer.exact_gp.optimizer,
            lr=config.trainer.exact_gp.lr,
            training_iters=config.trainer.exact_gp.training_iters,
            signal_variance=data.signal_variance,
            noise_variance_data=data.noise_variance,
            noise_variance_model=trainer.get_noise_variance(),
            lengthscale=trainer.get_lengthscale(),
            train_mll=train_mll,
            val_mll=val_mll,
            test_mll=test_mll,
            kendall_tau=tau,
            spearman=spearman,
            train_losses=train_losses,
            val_losses=val_losses,
            y_pred_train=y_pred_train,
            y_pred_val=y_pred_val,
            y_pred_test=y_pred_test,
            var_train=var_train,
            var_val=var_val,
            var_test=var_test,
            # Data fields for predictions.csv
            X_train=data.X_train.numpy(),
            X_val=data.X_val.numpy(),
            X_test=data.X_test.numpy(),
            y_true_train=data.Y_train_true.numpy(),
            y_true_val=data.Y_val_true.numpy(),
            y_true_test=data.Y_test_true.numpy(),
            y_noisy_train=data.Y_train_noisy.numpy(),
            y_noisy_val=data.Y_val_noisy.numpy(),
            y_noisy_test=data.Y_test_noisy.numpy(),
        )

        trainer.cleanup()
        return result

    except Exception as e:
        print(f"ERROR training ExactGP with {kernel_name}: {e}")
        import traceback
        traceback.print_exc()

        # Log failure
        failure = FailureRecord(
            gp_type="ExactGP",
            fitness_fn=fn_name,
            kernel_name=kernel_name,
            seed=config.experiment.seed,
            snr_data=config.data.snr,
            snr_model=config.model.snr_model,
            error_type=type(e).__name__,
            error_message=str(e)[:500],
        )
        collector.add_failure(failure)
        return None


def train_pairwisegp(data, kernel_name, config, device, fn_name, collector):
    """Train PairwiseGP model and return ModelResult."""
    try:
        # Build model
        model = PairwiseGPModel(
            kernel_name=kernel_name,
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

        # Create trainer
        trainer = PairwiseGPTrainer(
            model=model,
            training_iters=config.trainer.pairwise_gp.training_iters,
            lr=config.trainer.pairwise_gp.lr,
            optimizer_name=config.trainer.pairwise_gp.optimizer,
        )

        # Train
        train_losses, val_losses = trainer.train(data)

        # Compute metrics
        train_mll = trainer.compute_mll(
            data.X_train, data.Y_train_noisy,
            comparisons=data.comparisons_train,
            X_pairwise=data.X_train_pairwise,
        )
        val_mll = trainer.compute_mll(
            data.X_val, data.Y_val_noisy,
            comparisons=data.comparisons_val,
            X_pairwise=data.X_val_pairwise,
        )
        # For test, generate comparisons from true values
        test_mll = trainer.compute_mll(data.X_test, data.Y_test_true)

        # Predictions
        y_pred_train, var_train = trainer.predict(data.X_train)
        y_pred_val, var_val = trainer.predict(data.X_val)
        y_pred_test, var_test = trainer.predict(data.X_test)

        # Ranking metrics (on test set with true values)
        tau, _ = kendalltau(data.Y_test_true.numpy().flatten(), y_pred_test.flatten())
        spearman, _ = spearmanr(data.Y_test_true.numpy().flatten(), y_pred_test.flatten())

        result = ModelResult(
            gp_type="PairwiseGP",
            fitness_fn=fn_name,
            kernel_name=kernel_name,
            seed=config.experiment.seed,
            snr_data=config.data.snr,
            snr_model=config.model.snr_model,
            optimizer=config.trainer.pairwise_gp.optimizer,
            lr=config.trainer.pairwise_gp.lr,
            training_iters=config.trainer.pairwise_gp.training_iters,
            signal_variance=data.signal_variance,
            noise_variance_data=data.noise_variance,
            noise_variance_model=np.nan,  # PairwiseGP has no explicit noise
            lengthscale=trainer.get_lengthscale(),
            train_mll=train_mll,
            val_mll=val_mll,
            test_mll=test_mll,
            kendall_tau=tau,
            spearman=spearman,
            train_losses=train_losses,
            val_losses=val_losses,
            y_pred_train=y_pred_train,
            y_pred_val=y_pred_val,
            y_pred_test=y_pred_test,
            var_train=var_train,
            var_val=var_val,
            var_test=var_test,
            # Data fields for predictions.csv
            X_train=data.X_train.numpy(),
            X_val=data.X_val.numpy(),
            X_test=data.X_test.numpy(),
            y_true_train=data.Y_train_true.numpy(),
            y_true_val=data.Y_val_true.numpy(),
            y_true_test=data.Y_test_true.numpy(),
            y_noisy_train=data.Y_train_noisy.numpy(),
            y_noisy_val=data.Y_val_noisy.numpy(),
            y_noisy_test=data.Y_test_noisy.numpy(),
        )

        trainer.cleanup()
        return result

    except Exception as e:
        print(f"ERROR training PairwiseGP with {kernel_name}: {e}")
        import traceback
        traceback.print_exc()

        # Log failure
        failure = FailureRecord(
            gp_type="PairwiseGP",
            fitness_fn=fn_name,
            kernel_name=kernel_name,
            seed=config.experiment.seed,
            snr_data=config.data.snr,
            snr_model=config.model.snr_model,
            error_type=type(e).__name__,
            error_message=str(e)[:500],
        )
        collector.add_failure(failure)
        return None


if __name__ == "__main__":
    main()
