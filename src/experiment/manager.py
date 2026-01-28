"""
Experiment directory and metadata management.
Extracted from module_1.py lines 124-158.
"""
import os
import sys
import glob
import datetime

from .logger import Logger


class ExperimentManager:
    """Manages experiment directories, logging, and run IDs."""

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.output_dir = None
        self.today_str = None
        self.run_of_day = None
        self._logger = None

    def setup(self) -> str:
        """
        Create experiment directory and set up logging.

        Returns:
            Path to the output directory.
        """
        os.makedirs(self.base_dir, exist_ok=True)

        self.today_str = datetime.datetime.now().strftime("%d%m%y")
        existing_dirs = glob.glob(os.path.join(self.base_dir, f"experiments_{self.today_str}_*"))

        run_ids = []
        for d in existing_dirs:
            if os.path.isdir(d):
                try:
                    run_id = int(os.path.basename(d).split('_')[-1])
                    run_ids.append(run_id)
                except ValueError:
                    pass

        self.run_of_day = max(run_ids) + 1 if run_ids else 0
        self.output_dir = os.path.join(self.base_dir, f"experiments_{self.today_str}_{self.run_of_day}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logging (tee stdout to file)
        log_path = os.path.join(
            self.output_dir,
            f"experiment_{self.today_str}_{self.run_of_day}_log.txt",
        )
        self._logger = Logger(log_path)
        sys.stdout = self._logger

        # Save experiment ID for shell scripts (e.g., run_experiment.sh)
        with open(".last_experiment_id", "w") as f:
            f.write(f"{self.today_str}_{self.run_of_day}")

        print(f"--- Saving experiment CSVs to ./{self.output_dir} ---")
        return self.output_dir

    @property
    def experiment_id(self) -> str:
        """Return the experiment ID string (e.g., '270126_0')."""
        return f"{self.today_str}_{self.run_of_day}"

    @property
    def plots_dir(self) -> str:
        """Return path to the plots directory, creating it if needed."""
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        return plots_dir

    @staticmethod
    def find_experiment(base_dir: str, experiment_id: str = None) -> str:
        """
        Find an experiment directory by ID, or return the latest.

        Args:
            base_dir: Base experiments directory.
            experiment_id: Optional specific ID (e.g., '270126_0').

        Returns:
            Path to the experiment directory.
        """
        if experiment_id:
            target_dir = os.path.join(base_dir, f"experiments_{experiment_id}")
            if os.path.exists(target_dir):
                return target_dir
            raise FileNotFoundError(f"Experiment folder not found for ID '{experiment_id}'")

        # Find latest by modification time
        exp_dirs = glob.glob(os.path.join(base_dir, "experiments_*"))
        if not exp_dirs:
            raise FileNotFoundError(f"No experiment folders found in '{base_dir}'")
        return max(exp_dirs, key=os.path.getmtime)
