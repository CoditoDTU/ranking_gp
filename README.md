# Ranking GP Inference

This repository contains a framework for running Gaussian Process (GP) inference experiments, specifically comparing Pairwise GP and Exact GP models on various benchmark fitness functions.

## Repository Structure

```text
ranking_gp/
├── run_experiments.py           # Main experiment runner
├── run_visualization.py         # Standalone visualization (re-plot from saved data)
├── config.yaml                  # Configuration file for experiment parameters
├── environment.yaml             # Conda environment definition
├── module_1.py                  # Legacy monolithic script (deprecated)
├── src/
│   ├── config/
│   │   ├── experiment_config.py # ExperimentConfig dataclass, config loading
│   │   └── cli.py               # Argument parsers for both scripts
│   ├── experiment/
│   │   ├── logger.py            # Logger (tees stdout to log file)
│   │   ├── manager.py           # ExperimentManager (directories, run IDs)
│   │   └── results.py           # ResultsCollector, TrainingResult dataclass
│   ├── trainers/
│   │   ├── base.py              # BaseTrainer abstract class
│   │   ├── pairwise_gp.py      # PairwiseGPTrainer
│   │   └── exact_gp.py         # ExactGPTrainer
│   ├── visualization/
│   │   └── experiment_plots.py  # 3x2 grid plots (loss, fit, monotonicity)
│   ├── models.py                # GP model definitions (gpytorch / botorch)
│   ├── fitness_functions.py     # Benchmark test functions
│   ├── datatools.py             # Pairwise comparison utilities
│   ├── noise.py                 # Noise generation
│   └── solvers/
│       └── get_solvers.py       # Optimizer factory (Adam, SGD, AdamW, LBFGS)
└── unittest/
    └── test_fitness_functions.py
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/CoditoDTU/ranking_gp.git
    cd ranking_gp
    ```

2.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yaml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate gp_inference
    ```

## Usage

### Running Experiments

```bash
# Run with default config
python run_experiments.py

# Run with a specific config file
python run_experiments.py --config config.yaml

# Override seed and noise type
python run_experiments.py --seed 42 --noise_type gaussian

# Run without generating plots (results are still saved)
python run_experiments.py --no-plot

# Clear the aggregate summary before running
python run_experiments.py --clear_aggregate
```

All CLI flags for `run_experiments.py`:

| Flag | Description |
|---|---|
| `--config` | Path to config YAML (default: `config.yaml`) |
| `--seed` | Random seed (overrides config) |
| `--noise_type` | Noise type (overrides config) |
| `--pairwise_training_iters` | PairwiseGP training iterations |
| `--pairwise_lr` | PairwiseGP learning rate |
| `--pairwise_optimizer` | PairwiseGP optimizer (Adam, SGD, AdamW, LBFGS) |
| `--exact_training_iters` | ExactGP training iterations |
| `--exact_lr` | ExactGP learning rate |
| `--exact_optimizer` | ExactGP optimizer (Adam, SGD, AdamW, LBFGS) |
| `--clear_aggregate` | Clear existing aggregate summary before running |
| `--no-plot` | Skip plot generation after experiments |

### Re-plotting from Saved Results

`run_visualization.py` regenerates plots from previously saved experiment data, without re-running experiments:

```bash
# Plot the latest experiment
python run_visualization.py

# Plot a specific experiment by ID
python run_visualization.py --id 270126_0
```

## Configuration

Modify `config.yaml` to change experiment parameters:

```yaml
experiment:
  seed: 42
  fitness_functions:
    - ackley
    - gramacy_and_lee
    - cosines
    - levy
    - sphere
  kernel_names:
    - squared_exponential
    - matern_5_2
    - matern_3_2
    - exponential
  nsamples: 50
  noise: False
  dimension: 1

  pairwise_gp:
    training_iters: 1500
    lr: 0.01
    optimizer: Adam      # Options: Adam, SGD, AdamW, LBFGS

  exact_gp:
    training_iters: 500
    lr: 0.1
    optimizer: Adam
```

## Experiment Outputs

All results are saved in the `experiments/` directory.

```text
experiments/
├── experiments_DDMMYY_ID/
│   ├── summary_DDMMYY_ID.csv       # Metrics summary (NLL, Kendall tau, Spearman)
│   ├── predictions_DDMMYY_ID.csv   # Merged predictions for all models
│   ├── losses_DDMMYY_ID.json       # Training losses (for independent re-plotting)
│   ├── log_DDMMYY_ID.txt           # Stdout log
│   └── plots/                      # Generated PDF visualizations
│       ├── ackley_squared_exponential.pdf
│       └── ...
├── aggregate_summary.csv            # Results across all seeds/runs
└── aggregate_stats.csv              # Mean/std statistics across seeds
```

### Output Files

- **Summary CSV** (`summary_*.csv`): Metrics for every model trained in the run (GP type, fitness function, kernel, NLL, Kendall's tau, Spearman).
- **Predictions CSV** (`predictions_*.csv`): Train/test data, predictions, variances, and lengthscales for all models.
- **Losses JSON** (`losses_*.json`): Per-iteration training losses, enabling `run_visualization.py` to re-plot loss curves independently.
- **Plots** (`plots/`): 3x2 PDF grids per function/kernel showing training loss, function fit (1D), and monotonicity for both ExactGP and PairwiseGP.
- **Aggregate files**: `aggregate_summary.csv` accumulates results across seeds; `aggregate_stats.csv` reports mean/std when multiple seeds exist.
