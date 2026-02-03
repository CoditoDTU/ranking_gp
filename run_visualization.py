#!/usr/bin/env python
"""
Visualization runner for GP experiments.

Generates plots from saved experiment results without re-running experiments.

Usage:
    python run_visualization.py                     # Plot latest experiment
    python run_visualization.py --id 270126_0       # Plot specific experiment
    python run_visualization.py --all               # Plot all experiments
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import yaml

from src.config import create_visualization_parser, create_experiment_config
from src.experiment import ExperimentManager
from src.visualization import plot_experiment_grid


def load_experiment_config(data_dir, fallback_config_path):
    """Load config from the experiment's saved config, falling back to base config.

    Returns:
        ExperimentConfig instance with the correct parameters for this experiment.
    """
    config_files = glob.glob(os.path.join(data_dir, "config_*.yaml"))
    if config_files:
        config_path = os.path.abspath(config_files[0])
        return create_experiment_config(config_path)
    return create_experiment_config(fallback_config_path)


def visualize_experiment(data_dir, fallback_config_path):
    """Generate plots for a single experiment directory.

    Args:
        data_dir: Path to the experiment directory.
        fallback_config_path: Path to base config.yaml (used if no saved config found).

    Returns:
        True if plots were generated, False if required files were missing.
    """
    # Find and load predictions CSV
    pred_files = glob.glob(os.path.join(data_dir, "predictions_*.csv"))
    if not pred_files:
        print(f"  Skipping {data_dir}: no predictions CSV")
        return False

    predictions_file = pred_files[0]
    df = pd.read_csv(predictions_file)

    # Find and load summary CSV
    summary_files = glob.glob(os.path.join(data_dir, "summary_*.csv"))
    if not summary_files:
        print(f"  Skipping {data_dir}: no summary CSV")
        return False

    df_summary = pd.read_csv(summary_files[0])

    # Load config (prefer experiment's saved config for correct LR/dimension)
    config = load_experiment_config(data_dir, fallback_config_path)

    # Load training losses if available
    loss_files = glob.glob(os.path.join(data_dir, "losses_*.json"))
    if loss_files:
        with open(loss_files[0], 'r') as f:
            training_losses = json.load(f)
    else:
        training_losses = {}

    # Load validation losses if available
    val_loss_files = glob.glob(os.path.join(data_dir, "val_losses_*.json"))
    if val_loss_files:
        with open(val_loss_files[0], 'r') as f:
            validation_losses = json.load(f)
    else:
        validation_losses = {}

    # Reconstruct prediction_data from CSV data
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Group by experiment_name to rebuild the prediction data
    experiments = df['experiment_name'].unique()

    # Build prediction_data structure: {fn_name: {kernel_name: {model_name: {...}}}}
    prediction_data = {}

    for exp_name in experiments:
        # Parse experiment name: ModelName_FnName_DD_NoiseType_KernelName
        # e.g. "PairwiseGP_gramacy_and_lee_1D_none_squared_exponential"
        parts = exp_name.split('_')
        model_name = parts[0]
        # Find the dimension part (e.g., "1D") to split correctly
        dim_idx = next(i for i, p in enumerate(parts) if p.endswith('D'))
        fn_name = '_'.join(parts[1:dim_idx])
        # noise_type is at dim_idx+1, kernel_name is everything after that
        kernel_name = '_'.join(parts[dim_idx + 2:])

        df_exp = df[df['experiment_name'] == exp_name]
        df_train = df_exp[df_exp['fold'] == 0]
        df_test = df_exp[df_exp['fold'] == 1]

        # Get metrics from summary
        summary_row = df_summary[
            (df_summary['GP'] == model_name) &
            (df_summary['FitnessFn'] == fn_name) &
            (df_summary['kernel'] == kernel_name)
        ]

        tau = summary_row['kendal_tau'].values[0] if len(summary_row) > 0 else 0.0
        spearman_val = summary_row['spearman'].values[0] if len(summary_row) > 0 else 0.0
        test_nll = summary_row['test_nll'].values[0] if len(summary_row) > 0 else 0.0

        pred_dict = {
            'X_train': df_train['X'].values.reshape(-1, 1) if config.dimension == 1 else np.array(df_train['X'].tolist()),
            'Y_train': df_train['y_true'].values,
            'X_test': df_test['X'].values.reshape(-1, 1) if config.dimension == 1 else np.array(df_test['X'].tolist()),
            'Y_test': df_test['y_true'].values,
            'y_pred': df_test['y_pred'].values,
            'std': df_test['std'].values,
            'tau': tau,
            'spearman': spearman_val,
            'test_nll': test_nll,
        }

        prediction_data.setdefault(fn_name, {}).setdefault(kernel_name, {})[model_name] = pred_dict

    # Generate plots
    for fn_name in prediction_data:
        for kernel_name in prediction_data[fn_name]:
            pdf_path = plot_experiment_grid(
                fn_name, kernel_name,
                prediction_data[fn_name][kernel_name],
                training_losses.get(fn_name, {}).get(kernel_name, {}),
                validation_losses.get(fn_name, {}).get(kernel_name, {}),
                config.dimension,
                config.exact_gp.lr,
                config.pairwise_gp.lr,
                plots_dir,
            )
            if pdf_path:
                print(f"  Saved: {pdf_path}")

    return True


def main():
    parser = create_visualization_parser()
    args = parser.parse_args()

    BASE_DIR = "experiments"

    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory '{BASE_DIR}' does not exist.")
        sys.exit(1)

    if args.all:
        # Find all experiment directories
        exp_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "experiments_*")))
        exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]

        if not exp_dirs:
            print("No experiment directories found.")
            sys.exit(1)

        print(f"Found {len(exp_dirs)} experiment(s)\n")
        generated = 0
        for i, data_dir in enumerate(exp_dirs, 1):
            print(f"[{i}/{len(exp_dirs)}] {os.path.basename(data_dir)}")
            if visualize_experiment(data_dir, args.config):
                generated += 1

        print(f"\n--- Visualization Complete: {generated}/{len(exp_dirs)} experiments plotted ---")
    else:
        # Single experiment (by ID or latest)
        try:
            data_dir = ExperimentManager.find_experiment(BASE_DIR, args.id)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        print(f"Targeting experiment folder: {data_dir}")
        if not visualize_experiment(data_dir, args.config):
            print("Error: Required files missing in experiment directory.")
            sys.exit(1)

        print("\n--- Visualization Complete ---")


if __name__ == "__main__":
    main()
