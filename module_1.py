import os
# Fix for OMP: Error #15 (Multiple OpenMP runtimes linking)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import warnings
import random
import argparse
import datetime
import glob
import yaml
import torch
import numpy as np
import pandas as pd
import gpytorch
import linear_operator
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from botorch.models import PairwiseGP
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

# Increase default jitter for numerical stability
# linear_operator.settings.cholesky_jitter._global_float_value = 1e-4
# linear_operator.settings.cholesky_jitter._global_double_value = 1e-3
# linear_operator.settings.cholesky_jitter._global_half_value = 1e-2

# Add src to path so we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from fitness_functions import *
from models import FlexibleExactGPModel, build_kernel
from noise import add_noise
from datatools import get_comparisons, apply_sigmoid

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # Suppress GPyTorch warning about input matching stored training data
    warnings.filterwarnings("ignore", message="The input matches the stored training data")

    # --- 3. EXPERIMENT PARAMETERS ---
    # Load configuration
    parser = argparse.ArgumentParser(description="Run GP Experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exp_config = config['experiment']

    SEED = exp_config['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    FITNESS_FUNCTIONS = exp_config['fitness_functions']
    NOISE_TYPES = exp_config['noise_types']
    KERNEL_NAMES = exp_config['kernel_names']
    params = exp_config['noise_params']
    
    # Data and Training constants
    NSAMPLES = exp_config['nsamples']
    NOISE = exp_config['noise']
    D = exp_config['dimension']

    # PairwiseGP settings
    PAIRWISE_TRAINING_ITERS = exp_config['pairwise_gp']['training_iters']
    PAIRWISE_LR = exp_config['pairwise_gp']['lr']
    PAIRWISE_OPTIMIZER = exp_config['pairwise_gp']['optimizer']

    # ExactGP settings
    EXACT_TRAINING_ITERS = exp_config['exact_gp']['training_iters']
    EXACT_LR = exp_config['exact_gp']['lr']
    EXACT_OPTIMIZER = exp_config['exact_gp']['optimizer']

    # Helper function to get optimizer
    def get_optimizer(optimizer_name, parameters, lr):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(parameters, lr=lr)
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(parameters, lr=lr)
        elif optimizer_name == 'AdamW':
            return torch.optim.AdamW(parameters, lr=lr)
        elif optimizer_name == 'LBFGS':
            return torch.optim.LBFGS(parameters, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Define fixed sigmoid parameters
    SIGMOID_K = exp_config['sigmoid']['k']
    SIGMOID_X0 = exp_config['sigmoid']['x0']

    # --- 4. MAIN EXPERIMENT LOOP ---
    base_dir = "experiments"
    os.makedirs(base_dir, exist_ok=True)

    today_str = datetime.datetime.now().strftime("%d%m%y")
    existing_dirs = glob.glob(os.path.join(base_dir, f"experiments_{today_str}_*"))
    run_ids = []
    for d in existing_dirs:
        if os.path.isdir(d):
            try:
                run_id = int(os.path.basename(d).split('_')[-1])
                run_ids.append(run_id)
            except ValueError:
                pass

    run_of_day = max(run_ids) + 1 if run_ids else 0
    output_dir = os.path.join(base_dir, f"experiments_{today_str}_{run_of_day}")
    os.makedirs(output_dir, exist_ok=True)

    # Redirect stdout to a log file in the experiment directory
    sys.stdout = Logger(os.path.join(output_dir, f"experiment_{today_str}_{run_of_day}_log.txt"))

    # Save experiment ID to a file so run_experiment.sh can read it
    with open(".last_experiment_id", "w") as f:
        f.write(f"{today_str}_{run_of_day}")
    print(f"--- Saving experiment CSVs to ./{output_dir} ---")

    metadata_list = []
    experiment_id = 0

    # Store training losses for plotting: {fn_name: {kernel_name: {'ExactGP': [...], 'PairwiseGP': [...]}}}
    training_losses = {}

    # Store prediction data for comprehensive plots
    # Structure: {fn_name: {kernel_name: {'ExactGP': {...}, 'PairwiseGP': {...}}}}
    prediction_data = {}

    for fn_name in FITNESS_FUNCTIONS:
        print(f"\n======================================")
        print(f"Testing Fitness Function: {fn_name}")
        print(f"======================================")

        fitness_fn = fitness_function(base_fn_name=fn_name, dimension = D)
        

        '''
       1. We should sample the whole f(x) domain
       2. Cumulative regret: sum(f*(global optimun) - f(x)(current best estimate))
       3. Specify how are we sampling/ sampling regimes we are interested in
        '''
        # Generating common TESTING DATA:
        # Sample ~10000 points in a linspace (grid) over the whole domain
        total_test_points = 100
        grid_points_per_dim = int(total_test_points ** (1/D))
        X_test = fitness_fn.sample_grids(grid_points_per_dim)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
        Y_test = torch.Tensor(fitness_fn.output(X_test))

        NOISE_ITERATOR = ['none'] if NOISE == False else NOISE_TYPES # When there is not noise

        for noise_type in NOISE_ITERATOR:# Here we sample the train data either with or without noise
            #print(f"\n--- Noise Type: {noise_type} ---")

            # TRAINING DATA -- Independent of the Noise module

            X_train = (fitness_fn.sample_uniform(NSAMPLES, seed = SEED)) # We get NSAMPLES samples
            X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
            Y_train = torch.Tensor(fitness_fn.output(X_train)) # X_train cannot be a tensor for the bayeso function

            if NOISE == False:
                comparisons, X_train_pairwise = get_comparisons(Y_train, X=X_train_tensor) # For PairwiseGP
                print(f"\n--- No Noise ---")
                noise_level = 0.0 # Set noise level for metadata
                noise_type = 'none' # Ensure noise_type is set for metadata
            else:
                noise_level = params.get('g_std', 0.0)
                print(f"\n--- Noise Type: {noise_type} ---")
                Y_noisy = add_noise(Y_train, noise_type='gaussian', noise_params=params)
                comparisons, X_train_pairwise = get_comparisons(Y_noisy, X=X_train_tensor) # For PairwiseGP

            for kernel_name in KERNEL_NAMES:
                print(f"\n----- Kernel: {kernel_name} -----")

                # Construct data dicts here to ensure freshness and avoid modification issues
                if NOISE == False:
                    df1_train_data = {
                            'fold': 0,
                            'X': X_train.flatten() if D == 1 else X_train.tolist(),
                            'y_true': Y_train.flatten().numpy()
                        }
                    df1_test_data = {
                            'fold': 1,
                            'X': X_test.flatten() if D == 1 else X_test.tolist(),
                            'y_true': Y_test.flatten().numpy()
                        }
                else:
                    df1_train_data = {
                            'fold': 0,
                            'X': X_train.flatten() if D == 1 else X_train.tolist(),
                            'y_true': Y_train.flatten().numpy(),
                            'y_noisy': Y_noisy.flatten().numpy()
                        }
                    df1_test_data = {
                            'fold': 1,
                            'X': X_test.flatten() if D == 1 else X_test.tolist(),
                            'y_true': Y_test.flatten().numpy(),
                            'y_noisy': np.nan,
                        }

                ########## --- Run PairwiseGP ---################################################################
                # try:  # COMMENTED OUT FOR DEBUGGING - uncomment when done

                model_name = "PairwiseGP"
                print(f"Training {model_name}...")

                kernel_bt = build_kernel(kernel_name, D) # KERNEL
                model_bt = PairwiseGP(X_train_pairwise, comparisons, covar_module=kernel_bt, consolidate_atol=0.0)# GP
                mll_bt = PairwiseLaplaceMarginalLogLikelihood(model_bt.likelihood,model= model_bt) # MARGINAL Log likelihood
                optimizer_bt = get_optimizer(PAIRWISE_OPTIMIZER, model_bt.parameters(), PAIRWISE_LR) # Optimizer
                

                # Send to device
                model_bt.to(device)
                mll_bt.to(device)

                # Training:
                model_bt.train()
                losses = []

                for i in range(PAIRWISE_TRAINING_ITERS):

                    optimizer_bt.zero_grad() # zero grads
                    output = model_bt(model_bt.datapoints) # forward
                    loss = -mll_bt(output, model_bt.train_targets) # Loss calc
                    loss.backward() # Backprop

                    optimizer_bt.step() # Optimizer step
                    losses.append(loss.item())

                    if (i+1) % 20 == 0:
                        try:
                            ls = model_bt.covar_module.base_kernel.lengthscale.item()
                            print(f"  Iter {i+1} - Loss: {loss.item():.4f} - Lengthscale: {ls:.3f}")
                        except AttributeError:
                            print(f"  Iter {i+1} - Loss: {loss.item():.4f}")

                # Store PairwiseGP losses for plotting
                if fn_name not in training_losses:
                    training_losses[fn_name] = {}
                if kernel_name not in training_losses[fn_name]:
                    training_losses[fn_name][kernel_name] = {}
                training_losses[fn_name][kernel_name]['PairwiseGP'] = losses.copy()

                # Testing:
                model_bt.eval() # Eval mode

                # Train NLL is simply the last loss value
                train_nll_bt = losses[-1]

                with torch.no_grad():

                    # Predict on train points
                    posterior_train_bt = model_bt.posterior(X_train_tensor.to(device))
                    y_pred_train_bt = posterior_train_bt.mean.squeeze().cpu().numpy()
                    variance_train_bt = posterior_train_bt.variance.squeeze().detach().cpu().numpy()
                    std_train_bt = np.sqrt(variance_train_bt)

                    # Predict on test points
                    posterior_test_bt = model_bt.posterior(X_test_tensor.to(device))
                    y_pred_test_bt = posterior_test_bt.mean.squeeze().cpu().numpy()
                    variance_test_bt = posterior_test_bt.variance.squeeze().detach().cpu().numpy()
                    std_test_bt = np.sqrt(variance_test_bt)

                # Compute Proxy Test NLL for PairwiseGP
                # Create test comparisons from true test labels
                test_comparisons = get_comparisons(Y_test)

                # Create proxy model with test data structure
                proxy_kernel = build_kernel(kernel_name, D)
                model_proxy = PairwiseGP(X_test_tensor, test_comparisons, covar_module=proxy_kernel, consolidate_atol=0.0).double()

                # Copy learned hyperparameters from trained model
                model_proxy.covar_module.load_state_dict(model_bt.covar_module.state_dict())
                mll_proxy = PairwiseLaplaceMarginalLogLikelihood(model_proxy.likelihood, model_proxy)

                # Evaluate NLL in train mode
                model_proxy.train()
                with torch.no_grad():
                    output_proxy = model_proxy(model_proxy.datapoints)
                    test_nll_bt = -mll_proxy(output_proxy, model_proxy.train_targets).item()

                # Cleanup proxy model
                del model_proxy, mll_proxy, proxy_kernel

                with torch.no_grad():
                    ### Extract lengthscale
                    try:
                        ls_tensor = model_bt.covar_module.base_kernel.lengthscale
                        ls_bt = ls_tensor.detach().cpu().numpy().flatten()
                        if ls_bt.size == 1:
                            ls_bt = ls_bt.item()
                        else:
                            ls_bt = str(ls_bt.tolist())
                    except AttributeError:
                        ls_bt = np.nan

                # --- Build and Save df1 ---
                df1_train = pd.DataFrame(df1_train_data)
                df1_train['y_pred'] = y_pred_train_bt
                df1_train['variance'] = variance_train_bt
                df1_train['std'] = std_train_bt
                df1_train['lengthscale'] = ls_bt
                df1_test = pd.DataFrame(df1_test_data)
                df1_test['y_pred'] = y_pred_test_bt
                df1_test['variance'] = variance_test_bt
                df1_test['std'] = std_test_bt
                df1_test['lengthscale'] = ls_bt
                df1 = pd.concat([df1_train, df1_test]).reset_index(drop=True)

                df1_filename = f"{model_name}_{fn_name}_{D}D_{noise_type}_{kernel_name}.csv"
                df1_filepath = os.path.join(output_dir, df1_filename)
                df1.to_csv(df1_filepath, index=False)

                # --- Calculate metrics for df2 ---
                df1_test_only = df1[df1['fold'] == 1]
                tau, _ = kendalltau(df1_test_only['y_true'], df1_test_only['y_pred'])
                spearman, _ = spearmanr(df1_test_only['y_true'], df1_test_only['y_pred'])

                metadata_list.append({
                    "id": experiment_id, "GP": model_name, "FitnessFn": fn_name,
                    "dimension": D, "seed": SEED, "Noise_type": noise_type, "noise_level": noise_level,
                    "kernel": kernel_name, "train_nll": train_nll_bt, "test_nll": test_nll_bt,
                    "kendal_tau": tau, "spearman": spearman})
                experiment_id += 1

                # Store PairwiseGP prediction data for plotting
                if fn_name not in prediction_data:
                    prediction_data[fn_name] = {}
                if kernel_name not in prediction_data[fn_name]:
                    prediction_data[fn_name][kernel_name] = {}
                prediction_data[fn_name][kernel_name]['PairwiseGP'] = {
                    'X_train': X_train.copy(),
                    'Y_train': Y_train.numpy().copy(),
                    'X_test': X_test.copy(),
                    'Y_test': Y_test.numpy().copy(),
                    'y_pred': y_pred_test_bt.copy(),
                    'std': std_test_bt.copy(),
                    'tau': tau,
                    'spearman': spearman,
                    'test_nll': test_nll_bt
                }

                # except Exception as e:  # COMMENTED OUT FOR DEBUGGING - uncomment when done
                #     print(f"ERROR training {model_name} with {kernel_name}: {e}")
                #     import traceback
                #     traceback.print_exc()
                
                # Cleanup PairwiseGP memory to prevent crashes during inference
                if 'model_bt' in locals(): del model_bt
                if 'mll_bt' in locals(): del mll_bt
                if 'optimizer_bt' in locals(): del optimizer_bt
                if 'posterior_train_bt' in locals(): del posterior_train_bt
                if 'posterior_test_bt' in locals(): del posterior_test_bt
                if 'output' in locals(): del output
                if 'train_output' in locals(): del train_output
                if 'loss' in locals(): del loss
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                ######################## EXACT GP ###################################################################
                try:
                    model_name = "ExactGP"
                    print(f"Training {model_name}...")
                    kernel_reg = build_kernel(kernel_name, D) # KERNEL

                    X_train_reg = X_train_tensor.to(device, dtype=torch.double)
                    X_test_reg = X_test_tensor.to(device, dtype=torch.double)
                    Y_train_reg = Y_train.to(device, dtype=torch.double).squeeze(-1) if NOISE == False else Y_noisy.to(device, dtype=torch.double).squeeze(-1)

                    ll = GaussianLikelihood().to(device, dtype=torch.double) # Sent to device # Likelihood for ExactGP

                    model_reg = FlexibleExactGPModel(X_train_reg, Y_train_reg, ll, kernel_reg)
                    model_reg.to(device, dtype=torch.double) # Sent to device

                    mll_reg = gpytorch.mlls.ExactMarginalLogLikelihood(ll, model_reg) # mll
                    mll_reg.to(device, dtype=torch.double) # Sent to device

                    optimizer_reg = get_optimizer(EXACT_OPTIMIZER, model_reg.parameters(), EXACT_LR) # Optimizer

                    # Training
                    model_reg.train()
                    ll.train()

                    losses = []

                    for i in range(EXACT_TRAINING_ITERS):
                        optimizer_reg.zero_grad()
                        output = model_reg(X_train_reg)
                        loss = -mll_reg(output, Y_train_reg).sum()
                        loss.backward()
                        optimizer_reg.step()
                        losses.append(loss.item())

                        if (i+1) % 20 == 0:
                            try:
                                ls = model_reg.covar_module.base_kernel.lengthscale.item()
                                print(f"  Iter {i+1} - Loss: {loss.item():.4f} - Lengthscale: {ls:.3f}")
                            except AttributeError:
                                print(f"  Iter {i+1} - Loss: {loss.item():.4f}")

                    # Store ExactGP losses for plotting
                    training_losses[fn_name][kernel_name]['ExactGP'] = losses.copy()

                    model_reg.eval()
                    ll.eval()

                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        # Predict on train points
                        posterior_train_reg = model_reg(X_train_reg)
                        y_pred_train_reg = posterior_train_reg.mean.cpu().numpy()
                        variance_train_reg = posterior_train_reg.variance.cpu().numpy()
                        std_train_reg = np.sqrt(variance_train_reg)

                        # Predict on test points
                        posterior_test_reg = model_reg(X_test_reg)
                        y_pred_test_reg = posterior_test_reg.mean.cpu().numpy()
                        variance_test_reg = posterior_test_reg.variance.cpu().numpy()
                        std_test_reg = np.sqrt(variance_test_reg)

                        # Compute train NLL
                        train_output = model_reg(X_train_reg)
                        pred_dist_train = ll(train_output)
                        train_nll_reg = -pred_dist_train.log_prob(Y_train_reg).sum().item()

                        # Compute test NLL
                        pred_dist_test = ll(model_reg(X_test_reg))
                        test_nll_reg = -pred_dist_test.log_prob(Y_test.to(device, dtype=torch.double)).sum().item()

                        # Extract lengthscale
                        try:
                            ls_tensor = model_reg.covar_module.base_kernel.lengthscale
                            ls_reg = ls_tensor.detach().cpu().numpy().flatten()
                            if ls_reg.size == 1:
                                ls_reg = ls_reg.item()
                            else:
                                ls_reg = str(ls_reg.tolist())
                        except AttributeError:
                            ls_reg = np.nan

                    # --- Build and Save df1 ---

                    df1_train = pd.DataFrame(df1_train_data)
                    df1_train['y_pred'] = y_pred_train_reg
                    df1_train['variance'] = variance_train_reg
                    df1_train['std'] = std_train_reg
                    df1_train['lengthscale'] = ls_reg
                    df1_test = pd.DataFrame(df1_test_data)
                    df1_test['y_pred'] = y_pred_test_reg
                    df1_test['variance'] = variance_test_reg
                    df1_test['std'] = std_test_reg
                    df1_test['lengthscale'] = ls_reg

                    df1 = pd.concat([df1_train, df1_test]).reset_index(drop=True)

                    df1_filename = f"{model_name}_{fn_name}_{D}D_{noise_type}_{kernel_name}.csv"
                    df1_filepath = os.path.join(output_dir, df1_filename)
                    df1.to_csv(df1_filepath, index=False)

                    # --- Calculate metrics for df2 ---
                    df1_test_only = df1[df1['fold'] == 1]
                    tau, _ = kendalltau(df1_test_only['y_true'], df1_test_only['y_pred'])
                    spearman, _ = spearmanr(df1_test_only['y_true'], df1_test_only['y_pred'])

                    metadata_list.append({
                        "id": experiment_id, "GP": model_name, "FitnessFn": fn_name,
                        "dimension": D, "seed": SEED, "Noise_type": noise_type, "noise_level": noise_level,
                        "kernel": kernel_name, "train_nll": train_nll_reg, "test_nll": test_nll_reg,
                        "kendal_tau": tau, "spearman": spearman})
                    experiment_id += 1

                    # Store ExactGP prediction data for plotting
                    prediction_data[fn_name][kernel_name]['ExactGP'] = {
                        'X_train': X_train.copy(),
                        'Y_train': Y_train.numpy().copy(),
                        'X_test': X_test.copy(),
                        'Y_test': Y_test.numpy().copy(),
                        'y_pred': y_pred_test_reg.copy(),
                        'std': std_test_reg.copy(),
                        'tau': tau,
                        'spearman': spearman,
                        'test_nll': test_nll_reg
                    }

                except Exception as e:
                    print(f"ERROR training {model_name} with {kernel_name}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Cleanup ExactGP memory to prevent crashes during inference
                if 'model_reg' in locals(): del model_reg
                if 'mll_reg' in locals(): del mll_reg
                if 'optimizer_reg' in locals(): del optimizer_reg
                if 'll' in locals(): del ll
                if 'X_train_reg' in locals(): del X_train_reg
                if 'X_test_reg' in locals(): del X_test_reg
                if 'Y_train_reg' in locals(): del Y_train_reg
                if 'posterior_train_reg' in locals(): del posterior_train_reg
                if 'posterior_test_reg' in locals(): del posterior_test_reg
                if 'train_output' in locals(): del train_output
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 5. FINAL METADATA DATAFRAME (df2) ---
    print("\n--- Experiment Complete ---")
    df2 = pd.DataFrame(metadata_list)

    # --- Re-order and Display Results ---
    try:
        final_columns = [
            'id', 'GP', 'FitnessFn', 'dimension', 'seed', 'Noise_type', 'noise_level', 'kernel',
            'train_nll', 'test_nll', 'kendal_tau', 'spearman']
        df2_final = df2.reindex(columns=final_columns)

        df2_filepath = os.path.join(output_dir, f"summary_{today_str}_{run_of_day}.csv")
        df2_final.to_csv(df2_filepath, index=False)
        print(f"Experiment summary saved to {df2_filepath}")

        print("\n--- Experiment Summary (df2) ---")
        print(df2_final)

    except KeyError as e:
        print(f"\nError re-ordering DataFrame columns: {e}")
        print("Displaying default DataFrame:")
        print(df2)
    except Exception as e:
        print(f"\nAn error occurred creating the DataFrame: {e}")

    print("\n--- CSV Generation Complete ---")

    # --- 6. GENERATE COMPREHENSIVE PLOTS (3 rows x 2 columns) ---
    print("\n--- Generating Comprehensive Plots ---")

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for fn_name in prediction_data:
        kernels = list(prediction_data[fn_name].keys())

        for kernel_name in kernels:
            kernel_preds = prediction_data[fn_name][kernel_name]
            kernel_losses = training_losses.get(fn_name, {}).get(kernel_name, {})

            # Check if we have data
            has_exact = 'ExactGP' in kernel_preds
            has_pairwise = 'PairwiseGP' in kernel_preds

            if not (has_exact or has_pairwise):
                continue

            # Create 3x2 grid:
            # Row 0: Training Loss
            # Row 1: Function Fit
            # Row 2: Monotonicity (True vs Pred)
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle(f'{fn_name} - {kernel_name}', fontsize=14, fontweight='bold')

            # Get common data
            ref_gp = 'ExactGP' if has_exact else 'PairwiseGP'
            X_train = kernel_preds[ref_gp]['X_train']
            Y_train = kernel_preds[ref_gp]['Y_train']
            X_test = kernel_preds[ref_gp]['X_test']
            Y_test = kernel_preds[ref_gp]['Y_test']

            # For 1D case, flatten and sort for proper line plots
            if D == 1:
                X_test_flat = X_test.flatten()
                sort_idx = np.argsort(X_test_flat)
                X_test_sorted = X_test_flat[sort_idx]
                Y_test_sorted = Y_test.flatten()[sort_idx]
                X_train_flat = X_train.flatten()
                Y_train_flat = Y_train.flatten()

            # ===== ROW 0: Training Loss =====
            # Column 0: ExactGP Training Loss
            ax_loss_exact = axes[0, 0]
            if 'ExactGP' in kernel_losses:
                losses_exact = kernel_losses['ExactGP']
                ax_loss_exact.plot(range(1, len(losses_exact) + 1), losses_exact, 'b-', linewidth=1)
                ax_loss_exact.set_xlabel('Iteration')
                ax_loss_exact.set_ylabel('Loss (NLL)')
                ax_loss_exact.set_title(f'ExactGP: Training Loss (LR={EXACT_LR})')
                ax_loss_exact.grid(True, alpha=0.3)
            else:
                ax_loss_exact.text(0.5, 0.5, 'No ExactGP loss data', ha='center', va='center', transform=ax_loss_exact.transAxes)
                ax_loss_exact.set_title('ExactGP: Training Loss')

            # Column 1: PairwiseGP Training Loss
            ax_loss_pairwise = axes[0, 1]
            if 'PairwiseGP' in kernel_losses:
                losses_pairwise = kernel_losses['PairwiseGP']
                ax_loss_pairwise.plot(range(1, len(losses_pairwise) + 1), losses_pairwise, 'g-', linewidth=1)
                ax_loss_pairwise.set_xlabel('Iteration')
                ax_loss_pairwise.set_ylabel('Loss (NLL)')
                ax_loss_pairwise.set_title(f'PairwiseGP: Training Loss (LR={PAIRWISE_LR})')
                ax_loss_pairwise.grid(True, alpha=0.3)
            else:
                ax_loss_pairwise.text(0.5, 0.5, 'No PairwiseGP loss data', ha='center', va='center', transform=ax_loss_pairwise.transAxes)
                ax_loss_pairwise.set_title('PairwiseGP: Training Loss')

            # ===== ROW 1: Function Fit =====
            # Column 0: ExactGP Function Fit
            ax_fit_exact = axes[1, 0]
            if has_exact and D == 1:
                exact_data = kernel_preds['ExactGP']
                y_pred_exact = exact_data['y_pred'].flatten()[sort_idx]
                std_exact = exact_data['std'].flatten()[sort_idx]
                lower_exact = y_pred_exact - 2 * std_exact
                upper_exact = y_pred_exact + 2 * std_exact

                ax_fit_exact.plot(X_test_sorted, Y_test_sorted, 'k--', label="Ground Truth", linewidth=1.5)
                ax_fit_exact.plot(X_test_sorted, y_pred_exact, 'b-', label="ExactGP Mean", linewidth=1.5)
                ax_fit_exact.fill_between(X_test_sorted, lower_exact, upper_exact, color='b', alpha=0.2, label="95% CI")
                ax_fit_exact.scatter(X_train_flat, Y_train_flat, c='k', marker='x', s=30, label="Train Data", zorder=5)
                ax_fit_exact.set_xlabel('X')
                ax_fit_exact.set_ylabel('Y')
                ax_fit_exact.set_title(f"ExactGP: Function Fit (NLL={exact_data['test_nll']:.2f})")
                ax_fit_exact.legend(loc='best', fontsize=8)
                ax_fit_exact.grid(True, alpha=0.3)
            elif has_exact and D > 1:
                ax_fit_exact.text(0.5, 0.5, f'ExactGP\n(D={D}, plotting not supported)',
                        ha='center', va='center', transform=ax_fit_exact.transAxes, fontsize=12)
                ax_fit_exact.set_title("ExactGP: Function Fit")
            else:
                ax_fit_exact.text(0.5, 0.5, 'No ExactGP data', ha='center', va='center', transform=ax_fit_exact.transAxes)
                ax_fit_exact.set_title("ExactGP: Function Fit")

            # Column 1: PairwiseGP Function Fit
            ax_fit_pairwise = axes[1, 1]
            if has_pairwise and D == 1:
                pairwise_data = kernel_preds['PairwiseGP']
                y_pred_pairwise = pairwise_data['y_pred'].flatten()[sort_idx]
                std_pairwise = pairwise_data['std'].flatten()[sort_idx]
                lower_pairwise = y_pred_pairwise - 2 * std_pairwise
                upper_pairwise = y_pred_pairwise + 2 * std_pairwise

                ax_fit_pairwise.plot(X_test_sorted, y_pred_pairwise, 'g-', label="PairwiseGP Mean", linewidth=1.5)
                ax_fit_pairwise.fill_between(X_test_sorted, lower_pairwise, upper_pairwise, color='g', alpha=0.2, label="95% CI")
                ax_fit_pairwise.set_xlabel('X')
                ax_fit_pairwise.set_ylabel('Latent Utility')
                ax_fit_pairwise.set_title(f"PairwiseGP: Function Fit (NLL={pairwise_data['test_nll']:.2f})")
                ax_fit_pairwise.legend(loc='best', fontsize=8)
                ax_fit_pairwise.grid(True, alpha=0.3)
            elif has_pairwise and D > 1:
                ax_fit_pairwise.text(0.5, 0.5, f'PairwiseGP\n(D={D}, plotting not supported)',
                        ha='center', va='center', transform=ax_fit_pairwise.transAxes, fontsize=12)
                ax_fit_pairwise.set_title("PairwiseGP: Function Fit (Latent Scale)")
            else:
                ax_fit_pairwise.text(0.5, 0.5, 'No PairwiseGP data', ha='center', va='center', transform=ax_fit_pairwise.transAxes)
                ax_fit_pairwise.set_title("PairwiseGP: Function Fit")

            # ===== ROW 2: Monotonicity (True vs Predicted) =====
            # Column 0: ExactGP True vs Predicted
            ax_mono_exact = axes[2, 0]
            if has_exact:
                exact_data = kernel_preds['ExactGP']
                y_pred_exact_all = exact_data['y_pred'].flatten()
                std_exact_all = exact_data['std'].flatten()
                y_true = Y_test.flatten()

                ax_mono_exact.errorbar(y_pred_exact_all, y_true, xerr=2*std_exact_all, fmt='o', color='b', alpha=0.3, markersize=4)
                # Add ideal fit line
                lims = [min(ax_mono_exact.get_xlim()[0], ax_mono_exact.get_ylim()[0]),
                        max(ax_mono_exact.get_xlim()[1], ax_mono_exact.get_ylim()[1])]
                ax_mono_exact.plot(lims, lims, 'r--', alpha=0.75, label='Ideal Fit')
                ax_mono_exact.set_xlabel('Predicted Y')
                ax_mono_exact.set_ylabel('True Y')
                ax_mono_exact.set_title(f"ExactGP: Monotonicity\nTau={exact_data['tau']:.3f}, Spearman={exact_data['spearman']:.3f}")
                ax_mono_exact.legend(loc='best', fontsize=8)
                ax_mono_exact.grid(True, alpha=0.3)
            else:
                ax_mono_exact.text(0.5, 0.5, 'No ExactGP data', ha='center', va='center', transform=ax_mono_exact.transAxes)
                ax_mono_exact.set_title("ExactGP: Monotonicity")

            # Column 1: PairwiseGP True vs Predicted
            ax_mono_pairwise = axes[2, 1]
            if has_pairwise:
                pairwise_data = kernel_preds['PairwiseGP']
                y_pred_pairwise_all = pairwise_data['y_pred'].flatten()
                std_pairwise_all = pairwise_data['std'].flatten()
                y_true = Y_test.flatten()

                ax_mono_pairwise.errorbar(y_pred_pairwise_all, y_true, xerr=2*std_pairwise_all, fmt='o', color='g', alpha=0.3, markersize=4)
                ax_mono_pairwise.set_xlabel('Predicted Latent Utility')
                ax_mono_pairwise.set_ylabel('True Y')
                ax_mono_pairwise.set_title(f"PairwiseGP: Monotonicity\nTau={pairwise_data['tau']:.3f}, Spearman={pairwise_data['spearman']:.3f}")
                ax_mono_pairwise.grid(True, alpha=0.3)
            else:
                ax_mono_pairwise.text(0.5, 0.5, 'No PairwiseGP data', ha='center', va='center', transform=ax_mono_pairwise.transAxes)
                ax_mono_pairwise.set_title("PairwiseGP: Monotonicity")

            plt.tight_layout()

            # Save as PDF
            pdf_path = os.path.join(plots_dir, f"{fn_name}_{kernel_name}.pdf")
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {pdf_path}")


if __name__ == "__main__":
    main()