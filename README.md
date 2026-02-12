# Ranking GP Inference

This repository contains a framework for running Gaussian Process (GP) inference experiments, specifically comparing Pairwise GP and Exact GP models on various benchmark fitness functions.

## Repository Structure

```text
ranking_gp/
├── run_experiments.py           # Main experiment runner
├── run_grid_search.py           # Grid search over experiment parameters
├── aggregate_grid_search.py     # Aggregate results from grid search runs
├── run_visualization.py         # Standalone visualization (re-plot from saved data)
├── configs/
│   ├── config.yaml              # Base experiment configuration
│   ├── config_new.yaml          # New config with SNR-based noise
│   ├── grid_config.yaml         # Grid search parameter space
│   └── environment.yaml         # Conda environment definition
├── src/
│   ├── config/
│   │   ├── config.py            # Config dataclasses (ExperimentConfig, DataConfig, etc.)
│   │   ├── cli.py               # Argument parsers for experiment scripts
│   │   └── experiment_config.py # Legacy experiment config (deprecated)
│   ├── data/
│   │   ├── dataset.py           # ExperimentData class (sampling, noise, comparisons)
│   │   ├── fitness_functions.py # Benchmark test functions
│   │   └── comparisons.py       # Pairwise comparison utilities
│   ├── models/
│   │   ├── base.py              # BaseGPModel abstract class
│   │   ├── exact_gp.py          # ExactGPModel (gpytorch)
│   │   ├── pairwise_gp.py       # PairwiseGPModel (botorch)
│   │   └── kernels.py           # Kernel construction utilities
│   ├── trainers/
│   │   ├── base.py              # BaseTrainer abstract class
│   │   ├── exact_gp.py          # ExactGPTrainer
│   │   └── pairwise_gp.py       # PairwiseGPTrainer
│   ├── results/
│   │   ├── types.py             # ModelResult, FailureRecord dataclasses
│   │   └── collector.py         # ResultsCollector (aggregates and saves best models)
│   ├── visualization/
│   │   ├── experiment_plots.py  # Per-experiment 3x2 grid plots
│   │   ├── grid_search_plots.py # MLL vs SNR, comparison heatmaps
│   │   └── plots.py             # Legacy plots
│   ├── experiment/
│   │   ├── logger.py            # Logger (tees stdout to log file)
│   │   ├── manager.py           # ExperimentManager (directories, run IDs)
│   │   ├── progress.py          # Progress tracking
│   │   └── results.py           # Legacy results (deprecated)
│   └── solvers/
│       └── get_solvers.py       # Optimizer factory (Adam, SGD, AdamW, LBFGS)
├── module_1.py                  # Legacy monolithic script (deprecated)
├── deprecated/                  # Old plotting scripts
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
    conda env create -f configs/environment.yaml
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
python run_experiments.py --config configs/config_new.yaml

# Override seed and SNR
python run_experiments.py --seed 42 --snr 10.0

# Specify optimizer and learning rate
python run_experiments.py --optimizer Adam --lr 0.01

# Run quietly (log to file only)
python run_experiments.py --quiet
```

All CLI flags for `run_experiments.py`:

| Flag | Description |
|---|---|
| `--config` | Path to config YAML (default: `configs/config_new.yaml`) |
| `--seed` | Random seed (overrides config) |
| `--output_dir` | Output directory (overrides config) |
| `--snr` | Data SNR - signal-to-noise ratio (overrides config). Use `inf` for no noise. |
| `--snr_model` | Model SNR for priors (overrides config) |
| `--n_train` | Number of training samples (overrides config) |
| `--n_test` | Number of test samples (overrides config) |
| `--val_fraction` | Fraction of training data for validation (0.0-1.0) |
| `--fitness_function` | Single fitness function name (overrides config list) |
| `--dimension` | Input dimension (overrides config) |
| `--kernel` | Single kernel name (overrides config list) |
| `--optimizer` | Optimizer for both GP types (Adam, SGD, AdamW, LBFGS) |
| `--lr` | Learning rate for both GP types |
| `--training_iters` | Training iterations for both GP types |
| `--exact_training_iters` | ExactGP training iterations |
| `--exact_lr` | ExactGP learning rate |
| `--exact_optimizer` | ExactGP optimizer |
| `--pairwise_training_iters` | PairwiseGP training iterations |
| `--pairwise_lr` | PairwiseGP learning rate |
| `--pairwise_optimizer` | PairwiseGP optimizer |
| `--selection_criterion` | Model selection criterion (`val_mll`, `kendall_tau`, `spearman`) |
| `--quiet`, `-q` | Suppress terminal output (log to file only) |

### Grid Search

`run_grid_search.py` sweeps over experiment parameters by generating the Cartesian product of value lists defined in `grid_config.yaml`:

```bash
# Run grid search with default grid_config.yaml
python run_grid_search.py

# Preview all commands without executing
python run_grid_search.py --dry_run

# Run up to 4 experiments in parallel
python run_grid_search.py --max_parallel 4
```

All CLI flags for `run_grid_search.py`:

| Flag | Description |
|---|---|
| `--grid_config` | Path to grid search YAML (default: `configs/grid_config.yaml`) |
| `--config` | Path to base experiment config YAML |
| `--dry_run` | Print commands without executing |
| `--max_parallel` | Max concurrent experiments (default: 1 = sequential) |

#### Grid Config Format

```yaml
grid_search:
  # Seeds to sweep
  seed: [42, 123, 456]

  # SNR values (data noise level)
  snr: [1.0, 5.0, 10.0, 20.0, 50.0]

  # Optimizers
  optimizer: [Adam, LBFGS]

  # Training iterations
  pairwise_training_iters: [500, 1000, 1500]
  exact_training_iters: [250, 500, 1000]

  # Learning rates
  pairwise_lr: [0.001, 0.01]
  exact_lr: [0.01, 0.1]
```

### Aggregating Grid Search Results

After running a grid search, use `aggregate_grid_search.py` to collect and analyze results:

```bash
python aggregate_grid_search.py --grid_dir experiments/grid_20260209_143022
```

This creates:
- `aggregate_results.csv`: All best models from all runs
- `best_overall_per_fitness.csv`: Summary table of best model per fitness function
- `best_overall/`: Complete files for overall best per fitness function
- `plots/`: MLL vs SNR plots and comparison heatmaps

### Re-plotting from Saved Results

```bash
# Plot the latest experiment
python run_visualization.py

# Plot a specific experiment by ID
python run_visualization.py --id exp_20260207_143022

# Plot all experiments
python run_visualization.py --all
```

## Configuration

Modify `configs/config_new.yaml` to change experiment parameters:

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
  gp_types:
    - PairwiseGP
    - ExactGP

data:
  n_train: 50
  n_test: 100
  dimension: 1
  val_fraction: 0.2
  snr: 10.0  # Signal-to-noise ratio (inf = no noise)

model:
  snr_model: 10.0  # SNR for model priors

trainer:
  pairwise_gp:
    training_iters: 1500
    lr: 0.01
    optimizer: Adam
  exact_gp:
    training_iters: 500
    lr: 0.1
    optimizer: Adam

selection_criterion: val_mll  # Options: val_mll, kendall_tau, spearman
```

### SNR-based Noise

The framework uses Signal-to-Noise Ratio (SNR) to control noise levels:

```
noise_variance = signal_variance / SNR
```

Where `signal_variance` is computed from the training data. Higher SNR means less noise:
- `SNR = inf`: No noise
- `SNR = 100`: Very low noise
- `SNR = 10`: Moderate noise
- `SNR = 1`: High noise (signal = noise)

## Experiment Outputs

All results are saved in the `experiments/` directory.

### Single Experiment Output

```text
experiments/exp_20260209_143022/
├── best_models.json        # Best model per (gp_type, fitness_fn) for aggregation
├── summary.csv             # Metrics summary for all best models
├── failures.csv            # Failed experiments (if any)
└── models/                 # Per-model detailed outputs
    ├── ExactGP_ackley/
    │   ├── losses.json     # Training and validation losses
    │   ├── hyperparams.json # Kernel hyperparameters
    │   ├── metrics.json    # MLL, Kendall tau, Spearman
    │   └── predictions.csv # Predictions with X, y_true, y_noisy, y_pred, variance, std
    └── PairwiseGP_ackley/
        └── ...
```

### Grid Search Output

```text
experiments/grid_20260209_143022/
├── grid_config.yaml        # Copy of grid search config
├── runs/                   # Individual experiment runs
│   ├── exp_s42_snr10_Adam/
│   ├── exp_s42_snr20_Adam/
│   └── ...
├── aggregate_results.csv   # All best models from all runs
├── best_overall_per_fitness.csv  # Best model per fitness function
├── best_overall/           # Complete data for best models
│   ├── ackley/
│   │   ├── losses.json
│   │   ├── hyperparams.json
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   └── source.json     # Source experiment reference
│   └── ...
└── plots/
    ├── mll_vs_snr_ExactGP.pdf
    ├── mll_vs_snr_PairwiseGP.pdf
    ├── normalized_mll_vs_snr_ExactGP.pdf
    ├── normalized_mll_vs_snr_PairwiseGP.pdf
    ├── comparison_test_mll.pdf
    └── comparison_kendall_tau.pdf
```

### Predictions CSV Format

The `predictions.csv` file contains per-sample predictions with:

| Column | Description |
|---|---|
| `fold` | Data split: 0=train, 1=val, 2=test |
| `index` | Sample index within fold |
| `X` | Input value(s) |
| `y_true` | Ground truth output |
| `y_noisy` | Noisy observation |
| `y_pred` | Model prediction |
| `variance` | Predictive variance |
| `std` | Predictive standard deviation |

### Failures CSV Format

If any experiments fail (e.g., PSD matrix errors), they are logged in `failures.csv`:

| Column | Description |
|---|---|
| `gp_type` | ExactGP or PairwiseGP |
| `fitness_fn` | Fitness function name |
| `kernel_name` | Kernel type |
| `seed` | Random seed |
| `snr_data` | Data SNR |
| `snr_model` | Model SNR |
| `error_type` | Exception class name |
| `error_message` | Error details |
| `timestamp` | When the error occurred |