# Ranking GP Inference

This repository contains a framework for running Gaussian Process (GP) inference experiments, specifically comparing Pairwise GP and Exact GP models on various benchmark fitness functions.

## Repository Structure

```text
ranking_gp/
├── run_experiments.py           # Main experiment runner
├── run_grid_search.sh           # Grid search over seeds, noise_variance, optimizers
├── run_grid_search_ntrain.sh    # Grid search over n_train (training set size)
├── run_grid_search.py           # Python grid search runner (deprecated, use .sh)
├── aggregate_grid_search.py     # Aggregate results from grid search runs
├── run_visualization.py         # Standalone visualization (re-plot from saved data)
├── configs/
│   ├── config_new.yaml          # Main experiment configuration
│   ├── config.yaml              # Legacy config
│   ├── grid_search.yaml         # Grid search config (noise_variance sweep)
│   ├── grid_search_ntrain.yaml  # Grid search config (n_train sweep)
│   └── environment.yaml         # Conda environment definition
├── src/
│   ├── config/
│   │   ├── config.py            # Config dataclasses (ExperimentConfig, DataConfig, etc.)
│   │   ├── cli.py               # Argument parsers for experiment scripts
│   │   └── experiment_config.py # Legacy experiment config (deprecated)
│   ├── data/
│   │   ├── dataset.py           # ExperimentData class (sampling, noise, comparisons)
│   │   ├── fitness_functions.py # 10 benchmark test functions
│   │   └── comparisons.py       # Pairwise comparison utilities
│   ├── models/
│   │   ├── base.py              # BaseModelWrapper abstract class
│   │   ├── exact_gp.py          # ExactGPModel (gpytorch)
│   │   ├── pairwise_gp.py       # PairwiseGPModel (botorch)
│   │   └── kernels.py           # Kernel construction utilities
│   ├── trainers/
│   │   ├── base.py              # BaseTrainer abstract class
│   │   ├── exact_gp.py          # ExactGPTrainer (with early stopping)
│   │   └── pairwise_gp.py       # PairwiseGPTrainer (with early stopping)
│   ├── results/
│   │   ├── types.py             # ModelResult, FailureRecord dataclasses
│   │   └── collector.py         # ResultsCollector (aggregates and saves best models)
│   ├── visualization/
│   │   ├── experiment_plots.py  # Per-experiment 3x2 grid plots
│   │   ├── grid_search_plots.py # MLL vs noise_variance, MLL vs n_train plots
│   │   └── plots.py             # Legacy plots
│   ├── experiment/
│   │   ├── logger.py            # Logger (tees stdout to log file)
│   │   ├── manager.py           # ExperimentManager (directories, run IDs)
│   │   ├── progress.py          # Progress tracking
│   │   └── results.py           # Legacy results (deprecated)
│   └── solvers/
│       └── get_solvers.py       # Optimizer factory (Adam, SGD, AdamW, LBFGS)
├── sanity_check.py              # ExactGP sanity check script
├── sanity_check_pairwise.py     # PairwiseGP sanity check script
├── module_1.py                  # Legacy monolithic script (deprecated)
└── deprecated/                  # Old plotting scripts
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

# Override seed and noise variance
python run_experiments.py --seed 42 --noise_variance 0.3

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
| `--noise_variance` | Data noise variance (overrides config) |
| `--n_train` | Number of training samples (overrides config) |
| `--n_test` | Number of test samples (overrides config) |
| `--val_fraction` | Fraction of training data for validation (0.0-1.0) |
| `--fitness_function` | Single fitness function name (overrides config list) |
| `--dimension` | Input dimension (overrides config) |
| `--noise-model` | Model noise belief for ExactGP prior |
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

The recommended way to run grid searches is using the shell scripts, which provide logging, resume functionality, and automatic aggregation.

#### Noise Variance Grid Search

Sweep over seeds, noise variance values, and optimizers:

```bash
# Start new grid search
./run_grid_search.sh

# Use custom grid config
./run_grid_search.sh --grid_config configs/my_grid.yaml

# Resume interrupted grid search
./run_grid_search.sh --resume experiments/grid_20260220_143022

# Run quietly
./run_grid_search.sh --quiet
```

Grid config format (`configs/grid_search.yaml`):

```yaml
# Seeds for reproducibility
seeds: [0, 1, 2, 3, 4, 5]

# Noise variance values to sweep
noise_variance: [0.1, 0.2, 0.3, 0.4, 0.5]

# Optimizers to test
optimizers: ["Adam"]

# Base experiment config file
base_config: "configs/config_new.yaml"
```

#### N_train Grid Search

Sweep over training set sizes with fixed learning rates:

```bash
# Start new n_train grid search
./run_grid_search_ntrain.sh

# Resume interrupted grid search
./run_grid_search_ntrain.sh --resume experiments/grid_ntrain_20260222_224147
```

Grid config format (`configs/grid_search_ntrain.yaml`):

```yaml
# Seeds for reproducibility
seeds: [0, 1, 2, 3, 4, 5]

# Training set sizes to sweep
n_trains: [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

# Fixed learning rates
exact_lr: 0.5
pairwise_lr: 0.01

# Noise variance for data generation
noise_variance: 0.3

# Base experiment config file
base_config: "configs/config_new.yaml"
```

### Aggregating Grid Search Results

After running a grid search, use `aggregate_grid_search.py` to collect and analyze results:

```bash
python aggregate_grid_search.py --grid_dir experiments/grid_20260220_143022
```

This creates:
- `aggregate_results.csv`: All best models from all runs
- `best_overall_per_fitness.csv`: Summary table of best model per fitness function
- `best_overall/`: Complete files for overall best per fitness function
- `plots/`: MLL vs noise_variance/n_train plots and comparison heatmaps

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
# ============================================================
# DATA CONFIGURATION
# ============================================================
data:
  fitness_functions:
    - ackley
    - gramacy_and_lee
    - cosines
    - levy
    - sphere
    - sum_of_different_powers
    - power_sum
    - zakharov
    - dixon_price
    - michalewicz

  dimension: 1
  n_train: 50
  n_test: 100
  val_fraction: 0.2
  noise_variance: 0.5  # Noise variance for data generation

# ============================================================
# MODEL CONFIGURATION
# ============================================================
model:
  kernels:
    - squared_exponential
    - matern_5_2
    - matern_3_2
    - exponential

  # Model's prior belief about noise variance (for ExactGP)
  noise_variance_model: 0.1
  # Factors to multiply data noise_variance for model prior sweep
  noise_variance_model_factors: [0.5, 1.0, 2.0]

# ============================================================
# TRAINER CONFIGURATION
# ============================================================
trainer:
  exact_gp:
    training_iters: 500
    lr: 0.5
    optimizer: Adam
    # Early stopping configuration
    early_stopping: true
    patience: 40
    min_relative_delta: 0.001  # 0.1% improvement threshold
    check_interval: 10

  pairwise_gp:
    training_iters: 2500
    lr: 0.01
    optimizer: Adam
    early_stopping: true
    patience: 50
    min_relative_delta: 0.001
    check_interval: 10

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
experiment:
  seed: 42
  selection_criterion: val_mll  # Options: val_mll, kendall_tau, spearman
  output_dir: experiments/
```

### Noise Variance

The framework uses `noise_variance` to control noise levels in the generated data:

```
y_noisy = y_true + N(0, noise_variance)
```

The model can have a different belief about noise via `noise_variance_model`, allowing experiments where the model's prior differs from the true data noise.

### Early Stopping

Both ExactGP and PairwiseGP trainers support early stopping:

- **patience**: Number of iterations without improvement before stopping
- **min_relative_delta**: Minimum relative improvement threshold (e.g., 0.001 = 0.1%)
- **check_interval**: How often to check validation loss

When early stopping triggers, the model is restored to the best checkpoint found during training.

## Fitness Functions

The framework includes 10 benchmark functions from various categories:

| Category | Functions |
|---|---|
| Many Local Minima | Ackley, Levy |
| Bowl-Shaped | Sphere, Sum of Different Powers |
| Plate-Shaped | Power Sum, Zakharov |
| Valley-Shaped | Dixon-Price |
| Steep Ridges | Michalewicz |
| Other | Gramacy & Lee, Cosines |

## Experiment Outputs

All results are saved in the `experiments/` directory.

### Single Experiment Output

```text
experiments/exp_20260220_143022/
├── config.yaml         # Config used for this run
├── best_models.json    # Best model per (gp_type, fitness_fn) for aggregation
├── summary.csv         # Metrics summary for all best models
├── failures.csv        # Failed experiments (if any)
└── models/             # Per-model detailed outputs
    ├── ExactGP_1.0x_ackley/
    │   ├── losses.json     # Training and validation losses
    │   ├── hyperparams.json # Kernel hyperparameters
    │   ├── metrics.json    # MLL, Kendall tau, Spearman
    │   └── predictions.csv # Predictions with X, y_true, y_noisy, y_pred, variance
    ├── ExactGP_0.5x_ackley/  # Different noise_variance_model factor
    ├── PairwiseGP_ackley/
    └── ...
```

### Grid Search Output

```text
experiments/grid_20260220_143022/
├── grid_search_config.yaml  # Copy of grid search config
├── base_config.yaml         # Copy of base experiment config
├── grid_search.log          # Complete log of grid search
├── completed_experiments.txt # List of completed experiments (for resume)
├── runs/                    # Individual experiment runs
│   ├── exp_0_sigma_0.1_Adam/
│   ├── exp_0_sigma_0.2_Adam/
│   └── ...
├── aggregate_results.csv    # All best models from all runs
├── best_overall_per_fitness.csv  # Best model per fitness function
├── best_overall/            # Complete data for best models
└── plots/
    ├── mll_vs_noise_variance_ExactGP.pdf
    ├── mll_vs_noise_variance_PairwiseGP.pdf
    ├── comparison_test_mll.pdf
    └── comparison_kendall_tau.pdf
```

### Predictions CSV Format

The `predictions.csv` file contains per-sample predictions:

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
| `noise_variance` | Data noise variance |
| `noise_variance_model` | Model noise variance |
| `error_type` | Exception class name |
| `error_message` | Error details |
| `timestamp` | When the error occurred |

## License

See [LICENSE](LICENSE) for details.
