#!/bin/bash
# run_grid_search.sh
#
# Grid search over seeds, SNRs, and optimizers.
# Each combination runs a full experiment (all fitness functions × all kernels).
#
# Usage:
#   ./run_grid_search.sh                              # Start new grid search
#   ./run_grid_search.sh --grid_config my_grid.yaml   # Use custom grid config
#   ./run_grid_search.sh --resume experiments/grid_X  # Resume crashed grid search
#   ./run_grid_search.sh --quiet                      # Less verbose output

set -o pipefail

# Default values
GRID_CONFIG="configs/grid_search.yaml"
QUIET=""
RESUME_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --grid_config)
            GRID_CONFIG="$2"
            shift 2
            ;;
        --quiet)
            QUIET="--quiet"
            shift
            ;;
        --resume)
            RESUME_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_grid_search.sh [--grid_config FILE] [--quiet] [--resume DIR]"
            exit 1
            ;;
    esac
done

# Check if grid config exists
if [[ ! -f "$GRID_CONFIG" ]]; then
    echo "Error: Grid config not found: $GRID_CONFIG"
    echo "Create one or use --grid_config to specify a different file."
    exit 1
fi

# Parse YAML config using Python
read_yaml() {
    python3 << EOF
import yaml

with open("$GRID_CONFIG") as f:
    config = yaml.safe_load(f)

# Output as shell-compatible format
seeds = config.get('seeds', [42])
snrs = config.get('snrs', [10])
optimizers = config.get('optimizers', ['Adam'])
base_config = config.get('base_config', 'config_new.yaml')

# Convert to space-separated strings for bash arrays
print(f"SEEDS=({' '.join(str(s) for s in seeds)})")
print(f"SNRS=({' '.join(str(s) for s in snrs)})")
opt_str = ' '.join(f'"{o}"' for o in optimizers)
print(f"OPTIMIZERS=({opt_str})")
print(f'CONFIG="{base_config}"')
EOF
}

# Load config from YAML
eval "$(read_yaml)"

# Setup grid directory
if [[ -n "$RESUME_DIR" ]]; then
    # Resume mode
    if [[ ! -d "$RESUME_DIR" ]]; then
        echo "Error: Resume directory not found: $RESUME_DIR"
        exit 1
    fi
    GRID_DIR="$RESUME_DIR"
    echo "=========================================="
    echo "Resuming Grid Search"
    echo "=========================================="
    echo "Resuming from: $GRID_DIR"
else
    # New grid search
    GRID_DIR="experiments/grid_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$GRID_DIR"
    echo "=========================================="
    echo "Grid Search Starting"
    echo "=========================================="
    echo "Output directory: $GRID_DIR"
fi

# Setup logging - all output goes to both console and log file
LOG_FILE="$GRID_DIR/grid_search.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo "Grid config: $GRID_CONFIG"
echo "Seeds: ${SEEDS[*]}"
echo "SNRs: ${SNRS[*]}"
echo "Optimizers: ${OPTIMIZERS[*]}"
echo "Base config: $CONFIG"
echo "Started at: $(date)"
echo "=========================================="

# Status file to track completed experiments
STATUS_FILE="$GRID_DIR/completed_experiments.txt"
touch "$STATUS_FILE"

# Function to check if experiment is already completed
is_completed() {
    local exp_name="$1"
    grep -q "^${exp_name}$" "$STATUS_FILE" 2>/dev/null
}

# Function to mark experiment as completed
mark_completed() {
    local exp_name="$1"
    echo "$exp_name" >> "$STATUS_FILE"
}

# Save configs (only on new run)
if [[ -z "$RESUME_DIR" ]]; then
    # Copy the grid search config
    cp "$GRID_CONFIG" "$GRID_DIR/grid_search_config.yaml"
    echo "Saved grid config to: $GRID_DIR/grid_search_config.yaml"

    # Copy the base config file for reproducibility
    if [[ -f "$CONFIG" ]]; then
        cp "$CONFIG" "$GRID_DIR/base_config.yaml"
        echo "Saved base config to: $GRID_DIR/base_config.yaml"
    else
        echo "Warning: Base config file not found: $CONFIG"
    fi
fi

# Count total and completed experiments
TOTAL=$((${#SEEDS[@]} * ${#SNRS[@]} * ${#OPTIMIZERS[@]}))
COMPLETED_COUNT=$(wc -l < "$STATUS_FILE" | tr -d ' ')
CURRENT=0
SKIPPED=0
FAILED=0

echo ""
echo "Total experiments: $TOTAL"
echo "Already completed: $COMPLETED_COUNT"
echo "Remaining: $((TOTAL - COMPLETED_COUNT))"
echo ""

# Run experiments
for seed in "${SEEDS[@]}"; do
    for snr in "${SNRS[@]}"; do
        for opt in "${OPTIMIZERS[@]}"; do
            CURRENT=$((CURRENT + 1))
            exp_name="exp_s${seed}_snr${snr}_${opt}"

            # Check if already completed
            if is_completed "$exp_name"; then
                echo "[$CURRENT/$TOTAL] Skipping (already done): $exp_name"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            echo ""
            echo "[$CURRENT/$TOTAL] Running: $exp_name"
            echo "  Seed: $seed, SNR: $snr, Optimizer: $opt"
            echo "  Started at: $(date)"

            # Run experiment
            python run_experiments.py \
                --config "$CONFIG" \
                --seed "$seed" \
                --snr "$snr" \
                --optimizer "$opt" \
                --output_dir "$GRID_DIR/runs/$exp_name" \
                $QUIET

            if [ $? -ne 0 ]; then
                echo "  WARNING: Experiment $exp_name failed at $(date)"
                FAILED=$((FAILED + 1))
                # Don't mark as completed so it can be retried on resume
            else
                echo "  Completed: $exp_name at $(date)"
                mark_completed "$exp_name"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "Grid Search Complete"
echo "=========================================="
echo "Finished at: $(date)"
echo "Summary:"
echo "  Total: $TOTAL"
echo "  Skipped (already done): $SKIPPED"
echo "  Failed: $FAILED"
echo "  Newly completed: $((TOTAL - SKIPPED - FAILED))"
echo ""
echo "Running aggregation..."

# Aggregate results
python aggregate_grid_search.py --grid_dir "$GRID_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "Aggregation complete!"
    echo "Results saved to: $GRID_DIR"
    echo ""
    echo "Key files:"
    echo "  - $GRID_DIR/aggregate_results.csv"
    echo "  - $GRID_DIR/best_overall_per_fitness.csv"
    echo "  - $GRID_DIR/best_overall/"
    echo "  - $GRID_DIR/plots/"
    echo "  - $GRID_DIR/grid_search.log"
    echo "  - $GRID_DIR/base_config.yaml"
    echo "  - $GRID_DIR/grid_search_config.yaml"
else
    echo "WARNING: Aggregation failed!"
fi

echo ""
echo "To resume if interrupted: ./run_grid_search.sh --resume $GRID_DIR"