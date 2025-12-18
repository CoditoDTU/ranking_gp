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
KERNEL_NAMES = ['squared_exponential', 'matern_5_2', 'matern_3_2', 'exponential', 'linear']

# --- 2. Locate Experiment ---
parser = argparse.ArgumentParser(description="Generate 1D ExactGP X vs Y plots.")
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
    
    # Filter the summary to just this function's runs AND ExactGP
    df_summary_fn = df_summary[(df_summary['FitnessFn'] == fn_name) & (df_summary['GP'] == 'ExactGP')]

    if df_summary_fn.empty:
        continue

    # Check Dimension
    if 'dimension' in df_summary_fn.columns:
        dim = df_summary_fn.iloc[0]['dimension']
    else:
        dim = 1 # Default to 1 if missing, or handle error

    if int(dim) != 1:
        print(f"Skipping {fn_name} (Dimension {dim} != 1). This script only plots 1D functions.")
        continue

    print(f"\nGenerating 1D plot for: {fn_name}...")
    df1_pieces = []

    for _, row in df_summary_fn.iterrows():
        kernel_name = row['kernel']
        gp_name = row['GP']
        noise_type = row['Noise_type']

        filename = f"{gp_name}_{fn_name}_{dim}D_{noise_type}_{kernel_name}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        try:
            temp_df = pd.read_csv(filepath)
            # Filter for test data only
            if 'fold' in temp_df.columns:
                temp_df = temp_df[temp_df['fold'] == 1].copy()
            
            temp_df['Kernel'] = kernel_name
            df1_pieces.append(temp_df)

        except FileNotFoundError:
            print(f"  WARNING: Missing CSV file: {filename}")
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")

    if not df1_pieces:
        print(f"  No data found for {fn_name}, skipping plot.")
        continue

    plot_data = pd.concat(df1_pieces)
    
    # Melt to have y_true and y_pred in the same column for plotting
    plot_data_long = plot_data.melt(id_vars=['X', 'Kernel'], value_vars=['y_true', 'y_pred'], 
                                    var_name='Type', value_name='Y')

    present_kernels = [k for k in KERNEL_NAMES if k in plot_data['Kernel'].unique()]

    try:
        g = sns.relplot(
            data=plot_data_long,
            x='X', y='Y',
            hue='Type',
            row='Kernel',
            kind='scatter',
            height=3, aspect=2,
            row_order=present_kernels,
            alpha=0.7
        )
        
        g.fig.suptitle(f"ExactGP 1D Fit: '{fn_name}'", y=1.02, fontweight='bold')
        plot_filename = f"{fn_name}_1D_ExactGP_fit.pdf"
        plt.savefig(os.path.join(PLOT_DIR, plot_filename), bbox_inches='tight', dpi=150)
        print(f"  Plot saved to {os.path.join(PLOT_DIR, plot_filename)}")
        plt.close(g.fig)

    except Exception as e:
        print(f"  ERROR generating plot for {fn_name}: {e}")