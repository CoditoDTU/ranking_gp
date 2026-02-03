#!/usr/bin/env python
"""
Grid search runner for GP ranking experiments.

Reads a grid config YAML that specifies lists of values for each parameter,
generates the Cartesian product, and runs one experiment per combination
by invoking run_experiments.py as a subprocess with CLI overrides.

Usage:
    python run_grid_search.py                            # uses grid_config.yaml
    python run_grid_search.py --grid_config my_grid.yaml
    python run_grid_search.py --dry_run                  # print commands without executing
    python run_grid_search.py --max_parallel 4           # run up to 4 experiments in parallel
"""
import argparse
import glob
import itertools
import os
import subprocess
import sys
import time

import yaml


# Maps grid config keys to CLI flag names for run_experiments.py
GRID_KEY_TO_CLI_FLAG = {
    "fitness_function":        "--fitness_function",
    "nsamples":                "--nsamples",
    "g_std":                   "--g_std",
    "pairwise_training_iters": "--pairwise_training_iters",
    "exact_training_iters":    "--exact_training_iters",
    "pairwise_optimizer":      "--pairwise_optimizer",
    "exact_optimizer":         "--exact_optimizer",
    # Existing overrides that could also be swept:
    "seed":                    "--seed",
    "pairwise_lr":             "--pairwise_lr",
    "exact_lr":                "--exact_lr",
    "noise_type":              "--noise_type",
    "val_fraction":            "--val_fraction",
}


# Maps grid config keys to paths within saved config YAML
# e.g. "pairwise_lr" -> ("pairwise_gp", "lr") means config['experiment']['pairwise_gp']['lr']
GRID_KEY_TO_CONFIG_PATH = {
    "pairwise_training_iters": ("pairwise_gp", "training_iters"),
    "exact_training_iters":    ("exact_gp", "training_iters"),
    "pairwise_lr":             ("pairwise_gp", "lr"),
    "exact_lr":                ("exact_gp", "lr"),
    "pairwise_optimizer":      ("pairwise_gp", "optimizer"),
    "exact_optimizer":         ("exact_gp", "optimizer"),
    "fitness_function":        ("fitness_functions",),
    "nsamples":                ("nsamples",),
    "g_std":                   ("noise_params", "g_std"),
    "seed":                    ("seed",),
    "noise_type":              ("noise_types",),
    "val_fraction":            ("val_fraction",),
}


def _normalize(v):
    """Normalize a value for comparison (round floats to avoid precision issues)."""
    if isinstance(v, float):
        return round(v, 10)
    return v


def _combo_to_key(combo):
    """Convert a combination dict to a hashable tuple."""
    return tuple(sorted((k, _normalize(v)) for k, v in combo.items()))


def _extract_value(config_dict, path):
    """Extract a nested value from a config dict by path tuple."""
    val = config_dict
    for key in path:
        val = val[key]
    return val


def find_completed_combinations(experiments_dir, grid_keys):
    """
    Scan experiment directories for completed runs.

    A run is considered complete if its folder contains both
    a config_*.yaml and a summary_*.csv file. The config is parsed
    to extract the grid-relevant parameter values.

    Returns:
        Set of hashable combo keys (tuples of sorted (param, value) pairs).
    """
    completed = set()

    if not os.path.isdir(experiments_dir):
        return completed

    for dirname in os.listdir(experiments_dir):
        exp_dir = os.path.join(experiments_dir, dirname)
        if not os.path.isdir(exp_dir):
            continue

        config_files = glob.glob(os.path.join(exp_dir, "config_*.yaml"))
        summary_files = glob.glob(os.path.join(exp_dir, "summary_*.csv"))

        if not config_files or not summary_files:
            continue

        try:
            with open(config_files[0], 'r') as f:
                saved = yaml.safe_load(f)
            exp = saved.get('experiment', saved)
        except Exception:
            continue

        combo = {}
        try:
            for key in grid_keys:
                path = GRID_KEY_TO_CONFIG_PATH.get(key)
                if path is None:
                    continue
                val = _extract_value(exp, path)
                # List values (fitness_functions, noise_types) from single-override runs
                if key in ("fitness_function", "noise_type") and isinstance(val, list):
                    val = val[0] if len(val) == 1 else None
                    if val is None:
                        break
                combo[key] = _normalize(val)
            else:
                # Only add if all keys were extracted successfully
                completed.add(_combo_to_key(combo))
        except (KeyError, IndexError, TypeError):
            continue

    return completed


def load_grid_config(path: str) -> dict:
    """Load grid search parameter lists from YAML."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Grid config not found: {path}")
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    grid = raw.get('grid_search', {})
    if not grid:
        raise ValueError("Grid config must have a 'grid_search' key with at least one parameter list.")
    return grid


def build_combinations(grid: dict):
    """
    Generate the Cartesian product of all parameter lists.

    Returns:
        List of dicts, each mapping param name -> single value.
    """
    keys = sorted(grid.keys())
    value_lists = [grid[k] for k in keys]
    combinations = []
    for values in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, values)))
    return combinations


def build_command(combo: dict, base_config: str) -> list:
    """Build the subprocess command for a single grid point."""
    cmd = [sys.executable, "run_experiments.py",
           "--config", base_config,
           "--quiet"]

    for key, value in combo.items():
        flag = GRID_KEY_TO_CLI_FLAG.get(key)
        if flag is None:
            raise ValueError(
                f"Unknown grid parameter '{key}'. "
                f"Valid keys: {list(GRID_KEY_TO_CLI_FLAG.keys())}"
            )
        cmd.extend([flag, str(value)])

    return cmd


def run_sequential(commands, dry_run=False):
    """Run all commands sequentially."""
    total = len(commands)
    failed = []
    for i, cmd in enumerate(commands, 1):
        print(f"\n[{i}/{total}] {' '.join(cmd)}")
        if dry_run:
            continue
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            failed.append((i, cmd, result.returncode))
        else:
            print(f"  OK")
    return failed


def run_parallel(commands, max_parallel, dry_run=False):
    """Run commands with up to max_parallel concurrent subprocesses."""
    total = len(commands)
    if dry_run:
        for i, cmd in enumerate(commands, 1):
            print(f"[{i}/{total}] {' '.join(cmd)}")
        return []

    active = {}
    failed = []
    cmd_iter = iter(enumerate(commands, 1))
    launched = 0

    while launched < total or active:
        # Launch up to max_parallel
        while len(active) < max_parallel and launched < total:
            i, cmd = next(cmd_iter)
            print(f"[{i}/{total}] Launching: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            active[proc] = (i, cmd)
            launched += 1

        # Poll for completion
        finished = []
        for proc, (i, cmd) in active.items():
            retcode = proc.poll()
            if retcode is not None:
                finished.append(proc)
                if retcode != 0:
                    stderr = proc.stderr.read()[:500] if proc.stderr else ""
                    print(f"  [{i}] FAILED (exit code {retcode}): {stderr}")
                    failed.append((i, cmd, retcode))
                else:
                    print(f"  [{i}] OK")

        for proc in finished:
            del active[proc]

        if active and not finished:
            time.sleep(0.5)

    return failed


def main():
    parser = argparse.ArgumentParser(description="Grid search over GP experiment parameters")
    parser.add_argument("--grid_config", type=str, default="grid_config.yaml",
                        help="Path to grid search config YAML")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to base experiment config YAML")
    parser.add_argument("--clear_aggregate", action="store_true",
                        help="Clear aggregate CSV before first run")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--max_parallel", type=int, default=1,
                        help="Max parallel experiments (default: 1 = sequential)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip combinations that already have results in experiments/")
    args = parser.parse_args()

    # Load grid
    grid = load_grid_config(args.grid_config)
    combinations = build_combinations(grid)
    total = len(combinations)

    print(f"Grid search: {total} combinations from {len(grid)} parameters")
    for key, values in sorted(grid.items()):
        print(f"  {key}: {values}")

    # Resume: skip already-completed combinations
    if args.resume:
        completed = find_completed_combinations("experiments", list(grid.keys()))
        combinations = [
            combo for combo in combinations
            if _combo_to_key(combo) not in completed
        ]
        skipped = total - len(combinations)
        print(f"  Resume: {skipped} already completed, {len(combinations)} remaining")
        total = len(combinations)
        if total == 0:
            print("\nAll combinations already completed.")
            return

    print()

    # Build commands
    commands = []
    for i, combo in enumerate(combinations):
        cmd = build_command(combo, args.config)
        if i == 0 and args.clear_aggregate:
            cmd.append("--clear_aggregate")
        commands.append(cmd)

    # Run
    start = time.time()

    if args.max_parallel > 1:
        failed = run_parallel(commands, args.max_parallel, args.dry_run)
    else:
        failed = run_sequential(commands, args.dry_run)

    elapsed = time.time() - start

    print(f"\n{'='*50}")
    print(f"Grid search complete: {total - len(failed)}/{total} succeeded in {elapsed:.1f}s")
    if failed:
        print(f"Failed runs ({len(failed)}):")
        for i, cmd, code in failed:
            print(f"  [{i}] exit={code}: {' '.join(cmd)}")
    print(f"Results aggregated in: experiments/aggregate_summary.csv")


if __name__ == "__main__":
    main()
