"""
Results collection and storage.
Extracted from module_1.py lines 341-367, 495-524, 557-627.
"""
import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class TrainingResult:
    """Result from training a single GP model."""
    model_name: str           # 'PairwiseGP' or 'ExactGP'
    fn_name: str
    kernel_name: str
    noise_type: str
    dimension: int
    seed: int
    noise_level: float

    # Predictions
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    variance_train: np.ndarray
    variance_test: np.ndarray

    # Metrics
    train_nll: float
    test_nll: float
    kendall_tau: float
    spearman: float
    lengthscale: Any  # float or str (list) for ARD kernels

    # Training history
    losses: List[float] = field(default_factory=list)

    # Raw data references (for plotting)
    X_train: np.ndarray = None
    Y_train: np.ndarray = None
    X_test: np.ndarray = None
    Y_test: np.ndarray = None


class ResultsCollector:
    """Collects and saves experiment results."""

    def __init__(self, output_dir: str, experiment_id: str):
        self.output_dir = output_dir
        self.experiment_id = experiment_id

        self.metadata_list: List[Dict[str, Any]] = []
        self.all_predictions: List[pd.DataFrame] = []

        # For plotting: {fn_name: {kernel_name: {model_name: losses_list}}}
        self.training_losses: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        # For plotting: {fn_name: {kernel_name: {model_name: pred_dict}}}
        self.prediction_data: Dict[str, Dict[str, Dict[str, Dict]]] = {}

        self._experiment_counter = 0

    def add_result(self, result: TrainingResult,
                   df_train_data: Dict, df_test_data: Dict):
        """
        Add a training result to the collector.

        Args:
            result: TrainingResult from a trainer.
            df_train_data: Base training data dict (fold, X, y_true, ...).
            df_test_data: Base test data dict (fold, X, y_true, ...).
        """
        std_train = np.sqrt(result.variance_train)
        std_test = np.sqrt(result.variance_test)

        # Build prediction DataFrame
        df_train = pd.DataFrame(df_train_data)
        df_train['y_pred'] = result.y_pred_train
        df_train['variance'] = result.variance_train
        df_train['std'] = std_train
        df_train['lengthscale'] = result.lengthscale

        df_test = pd.DataFrame(df_test_data)
        df_test['y_pred'] = result.y_pred_test
        df_test['variance'] = result.variance_test
        df_test['std'] = std_test
        df_test['lengthscale'] = result.lengthscale

        df = pd.concat([df_train, df_test]).reset_index(drop=True)

        exp_name = (
            f"{result.model_name}_{result.fn_name}_{result.dimension}D"
            f"_{result.noise_type}_{result.kernel_name}"
        )
        df['experiment_name'] = exp_name
        self.all_predictions.append(df.copy())

        # Add metadata row
        self.metadata_list.append({
            "id": self._experiment_counter,
            "GP": result.model_name,
            "FitnessFn": result.fn_name,
            "dimension": result.dimension,
            "seed": result.seed,
            "Noise_type": result.noise_type,
            "noise_level": result.noise_level,
            "kernel": result.kernel_name,
            "train_nll": result.train_nll,
            "test_nll": result.test_nll,
            "kendal_tau": result.kendall_tau,
            "spearman": result.spearman,
        })
        self._experiment_counter += 1

        # Store losses for plotting
        fn = result.fn_name
        kernel = result.kernel_name
        model = result.model_name

        self.training_losses.setdefault(fn, {}).setdefault(kernel, {})[model] = result.losses.copy()

        # Store prediction data for plotting
        self.prediction_data.setdefault(fn, {}).setdefault(kernel, {})[model] = {
            'X_train': result.X_train,
            'Y_train': result.Y_train,
            'X_test': result.X_test,
            'Y_test': result.Y_test,
            'y_pred': result.y_pred_test,
            'std': std_test,
            'tau': result.kendall_tau,
            'spearman': result.spearman,
            'test_nll': result.test_nll,
        }

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_config(self, config) -> str:
        """Save the resolved experiment configuration as YAML for reproducibility."""
        filepath = os.path.join(self.output_dir, f"config_{self.experiment_id}.yaml")
        config_dict = asdict(config)
        with open(filepath, 'w') as f:
            yaml.dump({'experiment': config_dict}, f, default_flow_style=False, sort_keys=False)
        print(f"Experiment config saved to {filepath}")
        return filepath

    def save_losses(self) -> str:
        """Save training losses to JSON so they can be loaded independently."""
        filepath = os.path.join(self.output_dir, f"losses_{self.experiment_id}.json")
        with open(filepath, 'w') as f:
            json.dump(self.training_losses, f)
        print(f"Training losses saved to {filepath}")
        return filepath

    def save_predictions(self) -> Optional[str]:
        """Save merged predictions CSV. Returns filepath or None."""
        if not self.all_predictions:
            return None

        df = pd.concat(self.all_predictions, ignore_index=True)
        cols = ['experiment_name'] + [c for c in df.columns if c != 'experiment_name']
        df = df[cols]

        filepath = os.path.join(self.output_dir, f"predictions_{self.experiment_id}.csv")
        df.to_csv(filepath, index=False)
        print(f"Merged predictions saved to {filepath}")
        return filepath

    def save_summary(self) -> pd.DataFrame:
        """Save experiment summary CSV. Returns the DataFrame."""
        df = pd.DataFrame(self.metadata_list)

        final_columns = [
            'id', 'GP', 'FitnessFn', 'dimension', 'seed', 'Noise_type',
            'noise_level', 'kernel', 'train_nll', 'test_nll', 'kendal_tau', 'spearman',
        ]
        df_final = df.reindex(columns=final_columns)

        filepath = os.path.join(self.output_dir, f"summary_{self.experiment_id}.csv")
        df_final.to_csv(filepath, index=False)
        print(f"Experiment summary saved to {filepath}")
        print("\n--- Experiment Summary ---")
        print(df_final)

        return df_final

    def save_aggregate(self, base_dir: str, clear_existing: bool = False) -> Optional[pd.DataFrame]:
        """
        Update aggregate summary across multiple seeds/runs.

        Args:
            base_dir: Base experiments directory.
            clear_existing: Whether to clear the existing aggregate first.

        Returns:
            Aggregate stats DataFrame if multiple seeds exist, else None.
        """
        aggregate_path = os.path.join(base_dir, "aggregate_summary.csv")

        if clear_existing and os.path.exists(aggregate_path):
            os.remove(aggregate_path)
            print("Cleared existing aggregate summary.")

        df_current = pd.DataFrame(self.metadata_list)

        if os.path.exists(aggregate_path):
            existing = pd.read_csv(aggregate_path)
            updated = pd.concat([existing, df_current], ignore_index=True)
        else:
            updated = df_current.copy()

        updated.to_csv(aggregate_path, index=False)
        print(f"Aggregate summary (across seeds) saved to {aggregate_path}")

        # Generate per-experiment statistics if multiple seeds
        if len(updated['seed'].unique()) > 1:
            group_cols = ['GP', 'FitnessFn', 'dimension', 'Noise_type', 'noise_level', 'kernel']
            metric_cols = ['train_nll', 'test_nll', 'kendal_tau', 'spearman']

            agg_stats = updated.groupby(group_cols)[metric_cols].agg(['mean', 'std']).reset_index()
            agg_stats.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_stats.columns
            ]
            agg_stats['n_seeds'] = updated.groupby(group_cols)['seed'].nunique().values

            stats_path = os.path.join(base_dir, "aggregate_stats.csv")
            agg_stats.to_csv(stats_path, index=False)
            print(f"Aggregate statistics saved to {stats_path}")
            print("\n--- Aggregate Statistics (across seeds) ---")
            print(agg_stats)
            return agg_stats

        return None
