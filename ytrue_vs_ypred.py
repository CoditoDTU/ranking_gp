import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import sys
import warnings
import argparse

# --- 1. Configuration ---
BASE_DIR = "experiments"
# This list ensures row order consistency. 
# It filters dynamically based on what is actually found in the data.
KERNEL_NAMES = ['squared_exponential', 'matern_5_2', 'matern_3_2', 'exponential', 'linear']

# --- 2. Locate Experiment ---
parser = argparse.ArgumentParser(description="Generate plots for GP experiments.")
parser.add_argument("--id", type=str, help="Specific experiment ID (e.g., 181225_0) to plot. Defaults to latest.")
args = parser.parse_args()

if not os.path.exists(BASE_DIR):
    print(f"Error: Base directory '{BASE_DIR}' does not exist. Run module_1.py first.")
    sys.exit(1)

if args.id:
    # Try constructing the folder name assuming format experiments_ID
    target_dir = os.path.join(BASE_DIR, f"experiments_{args.id}")
    if not os.path.exists(target_dir):
        # Try assuming the user passed the full folder name
        target_dir = os.path.join(BASE_DIR, args.id)
        if not os.path.exists(target_dir):
             print(f"Error: Experiment folder not found for ID '{args.id}'.")
             sys.exit(1)
    print(f"Targeting specified experiment folder: {target_dir}")
    DATA_DIR = target_dir
else:
    # Find all experiment subfolders (e.g., experiments_181225_0)
    exp_dirs = glob.glob(os.path.join(BASE_DIR, "experiments_*"))
    if not exp_dirs:
        print(f"No experiment folders found in '{BASE_DIR}'.")
        sys.exit(1)
    # Sort by modification time to get the latest one
    latest_exp_dir = max(exp_dirs, key=os.path.getmtime)
    print(f"Targeting latest experiment folder: {latest_exp_dir}")
    DATA_DIR = latest_exp_dir

PLOT_DIR = os.path.join(DATA_DIR, "plots")

# Find the summary CSV inside the experiment folder
summary_files = glob.glob(os.path.join(DATA_DIR, "summary_*.csv"))
if not summary_files:
    print(f"Error: No summary CSV found in '{DATA_DIR}'.")
    sys.exit(1)

SUMMARY_FILE = summary_files[0]
print(f"Using summary file: {SUMMARY_FILE}")

# --- 3. Create Plot Directory ---
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Saving plots to {PLOT_DIR}")

# --- 4. Load Metadata (df2) ---
try:
    df_summary = pd.read_csv(SUMMARY_FILE)
except Exception as e:
    print(f"ERROR: Cannot read '{SUMMARY_FILE}': {e}")
    sys.exit(1)

if df_summary.empty:
    print("WARNING: Summary file is empty. No plots to generate.")
    sys.exit(0)

# Get a list of all fitness functions that were run
fitness_functions = df_summary['FitnessFn'].unique()
print(f"Found {len(fitness_functions)} functions to plot: {fitness_functions}")

# --- 5. Main Plotting Loop ---
for fn_name in fitness_functions:
    print(f"\nGenerating plot for: {fn_name}...")

    # --- 5a. Load all relevant df1 CSVs for this function ---
    df1_pieces = []

    # Filter the summary to just this function's runs
    df_summary_fn = df_summary[df_summary['FitnessFn'] == fn_name]

    if 'dimension' in df_summary_fn.columns:
        dim = df_summary_fn.iloc[0]['dimension']
    else:
        dim = "ND"

    for _, row in df_summary_fn.iterrows():
        # Reconstruct the exact filename
        kernel_name = row['kernel']
        gp_name = row['GP']
        noise_type = row['Noise_type']

        filename = f"{gp_name}_{fn_name}_{dim}D_{noise_type}_{kernel_name}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        try:
            # Load the individual CSV
            temp_df = pd.read_csv(filepath)

            # Filter for *only* test data
            if 'fold' in temp_df.columns:
                temp_df = temp_df[temp_df['fold'] == 1].copy()

            # Add the metadata to this dataframe
            temp_df['GP'] = gp_name
            temp_df['Kernel'] = kernel_name
            temp_df['kendal_tau'] = row['kendal_tau']

            df1_pieces.append(temp_df)

        except FileNotFoundError:
            print(f"  WARNING: Missing CSV file: {filename}")
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")

    if not df1_pieces:
        print(f"  No data found for {fn_name}, skipping plot.")
        continue

    # Combine all pieces for this function into one big DataFrame
    plot_data = pd.concat(df1_pieces)

    # Filter KERNEL_NAMES to only those present in data to avoid empty rows in plot
    present_kernels = [k for k in KERNEL_NAMES if k in plot_data['Kernel'].unique()]

    # --- 5b. Create the Grid Plot ---
    try:
        g = sns.relplot(
            data=plot_data,
            x='y_pred',
            y='y_true',
            col='GP',           # Columns are PairwiseGP vs ExactGP
            row='Kernel',       # Rows are the different kernels
            col_order=['PairwiseGP', 'ExactGP'], 
            row_order=present_kernels,
            hue='Kernel',       # Color-codes the points
            kind='scatter',
            height=4,
            aspect=1,
            facet_kws={'sharex': False, 'sharey': False},
            alpha=0.5,
            legend = False
        )

        # --- 5c. Customize Axes (Add y=x line and Tau) ---
        for (row_val, col_val), ax in g.axes_dict.items():
            # Add y=x line
            if ax.has_data():
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                lim_min = min(xlim[0], ylim[0])
                lim_max = max(xlim[1], ylim[1])
                ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.7)

            # Add Tau annotation
            tau_data = plot_data[(plot_data['Kernel'] == row_val) & (plot_data['GP'] == col_val)]
            if not tau_data.empty:
                tau = tau_data['kendal_tau'].iloc[0]
                ax.text(0.05, 0.95, f"τ = {tau:.3f}", transform=ax.transAxes, fontsize=10, fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5))

        # --- 5d. Set Titles and Save ---
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.set_axis_labels("y_pred (GP Inference)", "y_true (Ground Truth)")
        g.fig.suptitle(f"Model Performance: '{fn_name}' function", y=1.03, fontweight='bold', fontsize=16)

        plot_filename = f"{fn_name}_{dim}D_grid_ytrue_vs_y_pred.pdf"
        plot_filepath = os.path.join(PLOT_DIR, plot_filename)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.savefig(plot_filepath, bbox_inches='tight', dpi=150)
        
        print(f"  Plot saved to {plot_filepath}")
        plt.close(g.fig)

    except Exception as e:
        print(f"  ERROR generating plot for {fn_name}: {e}")
        import traceback
        traceback.print_exc()

print("\n--- All plotting complete! ---")