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
from scipy.stats import kendalltau
from botorch.models import PairwiseGP
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

# Add src to path so we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from fitness_functions import *
from models import FlexibleExactGPModel, build_kernel
from noise import add_noise
from datatools import get_comparisons, apply_sigmoid

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
    TRAINING_ITERS = exp_config['training_iters']
    LR = exp_config['lr']
    D = exp_config['dimension']

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

    # Save experiment ID to a file so run_experiment.sh can read it
    with open(".last_experiment_id", "w") as f:
        f.write(f"{today_str}_{run_of_day}")
    print(f"--- Saving experiment CSVs to ./{output_dir} ---")

    metadata_list = []
    experiment_id = 0

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
                comparisons = get_comparisons(Y_train) # For PairwiseGP
                print(f"\n--- No Noise ---")
                noise_level = 0.0 # Set noise level for metadata
                noise_type = 'none' # Ensure noise_type is set for metadata
            else:
                noise_level = params.get('g_std', 0.0)
                print(f"\n--- Noise Type: {noise_type} ---")
                Y_noisy = add_noise(Y_train, noise_type='gaussian', noise_params=params)
                comparisons = get_comparisons(Y_noisy) # For PairwiseGP

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
                try:
                    model_name = "PairwiseGP"
                    print(f"Training {model_name}...")

                    kernel_bt = build_kernel(kernel_name, D) # KERNEL
                    model_bt = PairwiseGP(X_train_tensor, comparisons, covar_module=kernel_bt)# GP
                    mll_bt = PairwiseLaplaceMarginalLogLikelihood(model_bt.likelihood,model= model_bt) # MARGINAL Log likelihood
                    optimizer_bt = torch.optim.Adam(model_bt.parameters(), lr=LR) # Optimizer

                    # Send to device
                    model_bt.to(device)
                    mll_bt.to(device)

                    # Training:
                    model_bt.train()
                    losses = []
                    # print(f"1. X_train_tensor (Input) device: {X_train_tensor.device}")
                    # try:
                    #     model_device = next(model_bt.parameters()).device
                    #     print(f"2. Model Parameters Device: {model_device}")
                    # except StopIteration:
                    #     print("2. Model has no parameters (requires setup before moving).")

                    # # --- FIX END ---
                    # print(f"4. Model's internal target (Y_train) device: {model_bt.train_targets.device}")

                    for i in range(TRAINING_ITERS):
                        optimizer_bt.zero_grad() # zero grads

                        output = model_bt(model_bt.unconsolidated_datapoints) # forward
                        loss = -mll_bt(output, model_bt.train_targets) # Loss calc
                        loss.backward() # Backprop

                        optimizer_bt.step() # Optimizer step
                        losses.append(loss.item())

                    # Testing:
                    model_bt.eval() # Eval mode

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

                        train_output = model_bt(model_bt.unconsolidated_datapoints)
                        train_nll = -mll_bt(train_output, model_bt.train_targets).item()

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

                    metadata_list.append({
                        "id": experiment_id, "GP": model_name, "FitnessFn": fn_name,
                        "dimension": D, "seed": SEED, "Noise_type": noise_type, "noise_level": noise_level,
                        "kernel": kernel_name, "nll": train_nll, "kendal_tau": tau})
                    experiment_id += 1

                except Exception as e:
                    print(f"ERROR training {model_name} with {kernel_name}: {e}")
                    import traceback
                    traceback.print_exc()
                
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

                    optimizer_reg = torch.optim.Adam(model_reg.parameters(), lr=LR) # Optimizer

                    # Training
                    model_reg.train()
                    ll.train()

                    losses = []
                    # print(f"1. X_train_reg (Input) device: {X_train_reg.device}")
                    # print(f"1.1 Y_train_reg (Input) device: {Y_train_reg.device}")
                    # print(f"1.1 Xtest_train_reg (Input) device: {X_test_tensor.device}") 
                    # try:
                    #     model_device = next(model_reg.parameters()).device
                    #     print(f"2. Model Parameters Device: {model_device}")
                    # except StopIteration:
                    #     print("2. Model has no parameters (requires setup before moving).")
                    # try:
                    #     likelihood_device = next(ll.parameters()).device
                    #     print(f"3. Likelihood (ll) device: {likelihood_device}")
                    # except StopIteration:
                    #     print("3. Likelihood device check failed (no parameters found).")
                    # # --- FIX END ---
                    # print(f"4. Model's internal target (Y_train) device: {model_reg.train_targets.device}")

                    for i in range(TRAINING_ITERS):
                        optimizer_reg.zero_grad() 
                        output = model_reg(X_train_reg)
                        loss = -mll_reg(output, Y_train_reg).sum()
                        loss.backward()
                        optimizer_reg.step()
                        losses.append(loss.item())

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

                        train_output = model_reg(X_train_reg) # logits
                        train_nll = -mll_reg(train_output, model_reg.train_targets).item() # nll for training after trainig
                        
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

                    metadata_list.append({
                        "id": experiment_id, "GP": model_name, "FitnessFn": fn_name,
                        "dimension": D, "seed": SEED, "Noise_type": noise_type, "noise_level": noise_level,
                        "kernel": kernel_name, "nll": train_nll, "kendal_tau": tau})
                    experiment_id += 1
                    
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
            'nll', 'kendal_tau']
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

if __name__ == "__main__":
    main()