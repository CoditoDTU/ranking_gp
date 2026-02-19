"""Visualization utilities for GP experiments."""
from .experiment_plots import plot_experiment_grid
from .plots import plot_fitness_function_grid, plot_from_saved_results
from .grid_search_plots import (
    normalize_mll_zscore,
    plot_mll_vs_snr,
    plot_mll_vs_ntrain,
    plot_mll_vs_noise_variance,
    plot_normalized_mll_vs_snr,
    plot_comparison_heatmap,
)

__all__ = [
    # Legacy interface
    'plot_experiment_grid',
    # New interface with ModelResult
    'plot_fitness_function_grid',
    'plot_from_saved_results',
    # Grid search plots
    'normalize_mll_zscore',
    'plot_mll_vs_snr',
    'plot_mll_vs_ntrain',
    'plot_mll_vs_noise_variance',
    'plot_normalized_mll_vs_snr',
    'plot_comparison_heatmap',
]
