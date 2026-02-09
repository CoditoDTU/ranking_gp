"""
Results collector for GP ranking experiments.

Collects results during training and saves only best models.
"""
from pathlib import Path
from typing import Dict, Tuple, List
import json
import numpy as np
import pandas as pd

from .types import ModelResult, FailureRecord
from ..config.config import SelectionCriterion


class ResultsCollector:
    """
    Collects results and saves only best models per (gp_type, fitness_fn).

    Usage:
        collector = ResultsCollector(output_dir, criterion=SelectionCriterion.VAL_MLL)

        # During experiment loop
        for result in results:
            collector.add_result(result)

        # After all training complete
        collector.save()
    """

    def __init__(
        self,
        output_dir: Path,
        criterion: SelectionCriterion = SelectionCriterion.VAL_MLL,
    ):
        """
        Initialize the collector.

        Args:
            output_dir: Directory to save results.
            criterion: Criterion for selecting best models.
        """
        self.output_dir = Path(output_dir)
        self.criterion = criterion
        self._all_results: Dict[Tuple[str, str], List[ModelResult]] = {}
        self._failures: List[FailureRecord] = []

    def add_result(self, result: ModelResult):
        """
        Add a result after training.

        Args:
            result: ModelResult from training.
        """
        key = (result.gp_type, result.fitness_fn)
        if key not in self._all_results:
            self._all_results[key] = []
        self._all_results[key].append(result)

    def add_failure(self, failure: FailureRecord):
        """
        Record a failed experiment.

        Args:
            failure: FailureRecord with error details.
        """
        self._failures.append(failure)

    def get_best_models(self) -> Dict[Tuple[str, str], ModelResult]:
        """
        Get best model per (gp_type, fitness_fn).

        Returns:
            Dictionary mapping (gp_type, fitness_fn) to best ModelResult.
        """
        best = {}
        for key, results in self._all_results.items():
            best[key] = min(results, key=lambda r: r.get_criterion_value(self.criterion))
        return best

    def save(self):
        """Save all outputs for best models only."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_models = self.get_best_models()

        self._save_best_models_json(best_models)
        self._save_summary_csv(best_models)
        self._save_individual_model_files(best_models)
        self._save_failures_csv()

    def _save_best_models_json(self, best_models: Dict[Tuple[str, str], ModelResult]):
        """Save best_models.json for grid search aggregation."""
        output = {}
        for (gp_type, fitness_fn), result in best_models.items():
            key = f"{gp_type}_{fitness_fn}"
            output[key] = {
                'gp_type': result.gp_type,
                'fitness_fn': result.fitness_fn,
                'kernel_name': result.kernel_name,
                'seed': result.seed,
                'snr_data': result.snr_data,
                'snr_model': result.snr_model,
                'optimizer': result.optimizer,
                'lr': result.lr,
                'training_iters': result.training_iters,
                'signal_variance': result.signal_variance,
                'noise_variance_data': result.noise_variance_data,
                'noise_variance_model': float(result.noise_variance_model) if not np.isnan(result.noise_variance_model) else None,
                'lengthscale': float(result.lengthscale) if not np.isnan(result.lengthscale) else None,
                'train_mll': result.train_mll,
                'val_mll': result.val_mll,
                'test_mll': result.test_mll,
                'kendall_tau': result.kendall_tau,
                'spearman': result.spearman,
            }

        with open(self.output_dir / "best_models.json", 'w') as f:
            json.dump(output, f, indent=2)

    def _save_summary_csv(self, best_models: Dict[Tuple[str, str], ModelResult]):
        """Save summary CSV with all best models."""
        rows = []
        for (gp_type, fitness_fn), result in best_models.items():
            rows.append(result.to_dict())

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "summary.csv", index=False)

    def _save_individual_model_files(self, best_models: Dict[Tuple[str, str], ModelResult]):
        """
        Save complete data for each best model.

        Creates directory structure:
        models/
        ├── ExactGP_ackley/
        │   ├── losses.json
        │   ├── hyperparams.json
        │   ├── metrics.json
        │   └── predictions.npz
        └── ...
        """
        models_dir = self.output_dir / "models"

        for (gp_type, fitness_fn), result in best_models.items():
            model_dir = models_dir / f"{gp_type}_{fitness_fn}"
            model_dir.mkdir(parents=True, exist_ok=True)

            # losses.json
            with open(model_dir / "losses.json", 'w') as f:
                json.dump({
                    'train_losses': result.train_losses,
                    'val_losses': result.val_losses,
                }, f, indent=2)

            # hyperparams.json
            with open(model_dir / "hyperparams.json", 'w') as f:
                json.dump({
                    'kernel': result.kernel_name,
                    'lr': result.lr,
                    'training_iters': result.training_iters,
                    'optimizer': result.optimizer,
                    'lengthscale': float(result.lengthscale) if not np.isnan(result.lengthscale) else None,
                    'noise_variance_model': float(result.noise_variance_model) if not np.isnan(result.noise_variance_model) else None,
                    'snr_data': result.snr_data,
                    'snr_model': result.snr_model,
                    'seed': result.seed,
                }, f, indent=2)

            # metrics.json
            with open(model_dir / "metrics.json", 'w') as f:
                json.dump({
                    'train_mll': result.train_mll,
                    'val_mll': result.val_mll,
                    'test_mll': result.test_mll,
                    'kendall_tau': result.kendall_tau,
                    'spearman': result.spearman,
                    'signal_variance': result.signal_variance,
                    'noise_variance_data': result.noise_variance_data,
                }, f, indent=2)

            # predictions.csv - one row per sample with fold info
            # fold: 0=train, 1=val, 2=test
            pred_rows = []

            def add_fold_rows(fold_num, X, y_true, y_noisy, y_pred, var):
                """Add rows for a single fold to pred_rows."""
                if y_pred is None:
                    return
                n = len(y_pred.flatten())
                y_pred_flat = y_pred.flatten()
                var_flat = var.flatten() if var is not None else np.full(n, np.nan)
                std_flat = np.sqrt(var_flat)

                # Handle X - convert to native Python floats
                if X is None:
                    X_vals = [np.nan] * n
                elif X.ndim == 1 or X.shape[1] == 1:
                    # 1D or single-feature: flatten to scalar per row
                    X_vals = X.flatten().tolist()
                else:
                    # Multi-D (>1 feature): list per row
                    X_vals = [row.tolist() for row in X]

                y_true_flat = y_true.flatten() if y_true is not None else np.full(n, np.nan)
                y_noisy_flat = y_noisy.flatten() if y_noisy is not None else np.full(n, np.nan)

                for i in range(n):
                    pred_rows.append({
                        'fold': fold_num,
                        'index': i,
                        'X': X_vals[i],
                        'y_true': float(y_true_flat[i]),
                        'y_noisy': float(y_noisy_flat[i]),
                        'y_pred': float(y_pred_flat[i]),
                        'variance': float(var_flat[i]),
                        'std': float(std_flat[i]),
                    })

            # Training predictions (fold=0)
            add_fold_rows(
                0, result.X_train, result.y_true_train, result.y_noisy_train,
                result.y_pred_train, result.var_train
            )

            # Validation predictions (fold=1)
            add_fold_rows(
                1, result.X_val, result.y_true_val, result.y_noisy_val,
                result.y_pred_val, result.var_val
            )

            # Test predictions (fold=2)
            add_fold_rows(
                2, result.X_test, result.y_true_test, result.y_noisy_test,
                result.y_pred_test, result.var_test
            )

            if pred_rows:
                pred_df = pd.DataFrame(pred_rows)
                pred_df.to_csv(model_dir / "predictions.csv", index=False)

    def _save_failures_csv(self):
        """Save failures.csv with all failed experiments."""
        if not self._failures:
            return

        rows = [f.to_dict() for f in self._failures]
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "failures.csv", index=False)

    def get_all_results_df(self) -> pd.DataFrame:
        """
        Get all results (not just best) as DataFrame.

        Useful for analysis of all kernel configurations.

        Returns:
            DataFrame with all results.
        """
        rows = []
        for key, results in self._all_results.items():
            for result in results:
                rows.append(result.to_dict())
        return pd.DataFrame(rows)
