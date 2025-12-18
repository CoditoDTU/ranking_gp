# Ranking GP Inference

This repository contains a framework for running Gaussian Process (GP) inference experiments, specifically comparing Pairwise GP and Exact GP models on various benchmark fitness functions.

## Repository Structure

```text
ranking_gp/
├── config.yaml              # Configuration file for experiment parameters
├── environment.yaml         # Conda environment definition
├── module_1.py              # Main training and inference script
├── plot_1d_exactgp.py       # Plotting script for 1D ExactGP fits
├── run_experiment.sh        # Pipeline script to run training and plotting sequentially
├── ytrue_vs_ypred.py        # Plotting script for Ground Truth vs Prediction grids
├── src/                     # Source code for models and utilities
│   ├── fitness_functions.py
│   ├── models.py
│   └── ...
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/CoditoDTU/ranking_gp.git
    cd ranking_gp
    ```

2.  **Create the Conda environment:**
    This project uses a `environment.yaml` file to manage dependencies (PyTorch, GPyTorch, BoTorch, etc.).
    ```bash
    conda env create -f environment.yaml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate gp_inference
    ```

## Usage

### Running the Full Pipeline
The easiest way to run an experiment and generate plots is using the shell script:

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

This script will:
1.  Run `module_1.py` to train the gps based on `config.yaml`.
2.  Automatically detect the generated experiment ID.
3.  Run `ytrue_vs_ypred.py` to generate prediction vs. ground truth plots.
4.  Run `plot_1d_exactgp.py` (if dimension is 1) to visualize the fit.

### Running Scripts Individually

**1. Training (`module_1.py`)**
Runs the experiment. Results are saved in `experiments/experiments_DATE_ID/`.
```bash
python module_1.py --config config.yaml
```

**2. Plotting (`ytrue_vs_ypred.py`)**
Generates "Ground Truth vs Prediction" grid plots.
```bash
# Plot the latest experiment
python ytrue_vs_ypred.py

# Plot a specific experiment ID
python ytrue_vs_ypred.py --id 181225_0
```

## Configuration
Modify `config.yaml` to change experiment parameters such as `fitness_functions`, `dimension`, `noise`, `kernel_names`, and `training_iters`.
