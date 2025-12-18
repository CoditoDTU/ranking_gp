#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Experiment (module_1.py)..."
# Pass arguments (like --config) to the training script
python module_1.py "$@"

if [ -f .last_experiment_id ]; then
    EXP_ID=$(cat .last_experiment_id)
    echo "Experiment finished. Generating plots (ytrue_vs_ypred.py) for ID: $EXP_ID..."
    python ytrue_vs_ypred.py --id "$EXP_ID"

    echo "Generating 1D ExactGP plots (plot_1d_exactgp.py) for ID: $EXP_ID..."
    python plot_1d_exactgp.py --id "$EXP_ID"
else
    echo "Experiment finished. Generating plots (ytrue_vs_ypred.py) for latest..."
    python ytrue_vs_ypred.py
    echo "Generating 1D ExactGP plots (plot_1d_exactgp.py) for latest..."
    python plot_1d_exactgp.py
fi

echo "Pipeline Complete!"