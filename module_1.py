import os
import sys
import random
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

from fitness_functions import fitness_function
from models import FlexibleExactGPModel, build_kernel
from noise import add_noise
from datatools import get_comparisons, apply_sigmoid

def main():
    # --- 3. EXPERIMENT PARAMETERS ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    FITNESS_FUNCTIONS = ['ackley', 'gramacy_and_lee', 'cosines']
    NOISE_TYPES = ['gaussian'] # Add 'poisson' if you've configured noise_params
    KERNEL_NAMES = ['squared_exponential', 'matern_5_2', 'matern_3_2', 'exponential']#, 'linear']#, 'periodic']
    GP_TYPES = ['PairwiseGP', 'ExactGP']
    params = {'g_std' : 1.1, 'h_std': 0.1 , 'amplitude_factor': 0.5  } # Params for GP noise
    
    # Data and Training constants
    NSAMPLES = 96
    NOISE = False
    TRAINING_ITERS = 200
    LR = 0.1
    D = 1 # Dimension (used by BayesoFitnessFunction)

    # Define fixed sigmoid parameters
    SIGMOID_K = 2.0
    SIGMOID_X0 = 5.0

    # --- 4. MAIN EXPERIMENT LOOP ---
    output_dir = "experiments"
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Saving experiment CSVs to ./{output_dir} ---")

    metadata_list = []
    experiment_id = 0

    for fn_name in FITNESS_FUNCTIONS:
        print(f"\n======================================")
        print(f"Testing Fitness Function: {fn_name}")
        print(f"======================================")

        fitness_fn = fitness_function(base_fn_name=fn_name, dimension = D)

        # Generating common TESTING DATA:
        X_test = fitness_fn.sample_uniform(100, seed = SEED + random.randint(0,100)) # Respecting fitness functions boundaries
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

                df1_train_data = {
                        'fold': 0,
                        'X': X_train.flatten(),
                        'y_true': Y_train.flatten().numpy()
                    }

                df1_test_data = {
                        'fold': 1,
                        'X': X_test.flatten(),
                        'y_true': Y_test.flatten().numpy()
                    }
            else:
                noise_level = params.get('g_std', 0.0)
                print(f"\n--- Noise Type: {noise_type} ---")
                Y_noisy = add_noise(Y_train, noise_type='gaussian', noise_params=params)
                comparisons = get_comparisons(Y_noisy) # For PairwiseGP

                df1_train_data = {
                        'fold': 0,
                        'X': X_train.flatten(),
                        'y_true': Y_train.flatten().numpy(),
                        'y_noisy': Y_noisy.flatten().numpy()
                    }

                df1_test_data = {
                        'fold': 1,
                        'X': X_test.flatten(),
                        'y_true': Y_test.flatten().numpy(),
                        'y_noisy': np.nan,
                    }

            for kernel_name in KERNEL_NAMES:
                print(f"\n----- Kernel: {kernel_name} -----")

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
                    print(f"1. X_train_tensor (Input) device: {X_train_tensor.device}")
                    try:
                        model_device = next(model_bt.parameters()).device
                        print(f"2. Model Parameters Device: {model_device}")
                    except StopIteration:
                        print("2. Model has no parameters (requires setup before moving).")

                    # --- FIX END ---
                    print(f"4. Model's internal target (Y_train) device: {model_bt.train_targets.device}")

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
                        y_pred_train_bt = model_bt.posterior(X_train_tensor.to(device)).mean.squeeze().cpu().numpy() # Logits

                        # Predict on test points (X_test)
                        y_pred_test_bt = model_bt.posterior(X_test_tensor.to(device)).mean.cpu().numpy()
                        train_output = model_bt(model_bt.unconsolidated_datapoints)
                        train_nll = -mll_bt(train_output, model_bt.train_targets).item()

                    # --- Build and Save df1 ---
                    df1_train = pd.DataFrame(df1_train_data)
                    df1_train['y_pred'] = y_pred_train_bt
                    df1_test = pd.DataFrame(df1_test_data)
                    df1_test['y_pred'] = y_pred_test_bt
                    df1 = pd.concat([df1_train, df1_test]).reset_index(drop=True)
                    df1['y_true_sigmoid'] = apply_sigmoid(df1['y_true'], k=SIGMOID_K, x0=SIGMOID_X0).numpy()
                    df1['y_pred_sigmoid'] = apply_sigmoid(df1['y_pred'], k=SIGMOID_K, x0=SIGMOID_X0).numpy()

                    if NOISE == True:
                        df1['y_noisy_sigmoid'] = apply_sigmoid(df1['y_noisy'], k=SIGMOID_K, x0=SIGMOID_X0).numpy()

                    df1_filename = f"{model_name}_{fn_name}_{noise_type}_{kernel_name}.csv"
                    df1_filepath = os.path.join(output_dir, df1_filename)
                    df1.to_csv(df1_filepath, index=False)
                    
                    # --- Calculate metrics for df2 ---
                    df1_test_only = df1[df1['fold'] == 1]
                    tau, _ = kendalltau(df1_test_only['y_true'], df1_test_only['y_pred'])
                    tau_sig, _ = kendalltau(df1_test_only['y_true_sigmoid'], df1_test_only['y_pred_sigmoid'])

                    metadata_list.append({
                        "id": experiment_id, "GP": model_name, "FitnessFn": fn_name,
                        "seed": SEED, "Noise_type": noise_type, "noise_level": noise_level,
                        "kernel": kernel_name, "nll": train_nll, "kendal_tau": tau,
                        "kendal_tau_sigmoid": tau_sig
                        })
                    experiment_id += 1

                except Exception as e:
                    print(f"ERROR training {model_name} with {kernel_name}: {e}")
                    import traceback
                    traceback.print_exc()

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
                    print(f"1. X_train_reg (Input) device: {X_train_reg.device}")
                    print(f"1.1 Y_train_reg (Input) device: {Y_train_reg.device}")
                    print(f"1.1 Xtest_train_reg (Input) device: {X_test_tensor.device}") 
                    try:
                        model_device = next(model_reg.parameters()).device
                        print(f"2. Model Parameters Device: {model_device}")
                    except StopIteration:
                        print("2. Model has no parameters (requires setup before moving).")
                    try:
                        likelihood_device = next(ll.parameters()).device
                        print(f"3. Likelihood (ll) device: {likelihood_device}")
                    except StopIteration:
                        print("3. Likelihood device check failed (no parameters found).")
                    # --- FIX END ---
                    print(f"4. Model's internal target (Y_train) device: {model_reg.train_targets.device}")

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
                        y_pred_train_reg = model_reg(X_train_reg).mean.cpu().numpy()
                        y_pred_test_reg = model_reg(X_test_reg).mean.cpu().numpy()

                        train_output = model_reg(X_train_reg) # logits
                        train_nll = -mll_reg(train_output, model_reg.train_targets).item() # nll for training after trainig
                    # --- Build and Save df1 ---

                    df1_train = pd.DataFrame(df1_train_data)
                    df1_train['y_pred'] = y_pred_train_reg

                    df1_test = pd.DataFrame(df1_test_data)
                    df1_test['y_pred'] = y_pred_test_reg

                    df1 = pd.concat([df1_train, df1_test]).reset_index(drop=True)
                    df1['y_true_sigmoid'] = apply_sigmoid(df1['y_true'], k=SIGMOID_K, x0=SIGMOID_X0).numpy()
                    df1['y_pred_sigmoid'] = apply_sigmoid(df1['y_pred'], k=SIGMOID_K, x0=SIGMOID_X0).numpy()
                    if NOISE == True:
                        df1['y_noisy_sigmoid'] = apply_sigmoid(df1['y_noisy'], k=SIGMOID_K, x0=SIGMOID_X0).numpy()


                    df1_filename = f"{model_name}_{fn_name}_{noise_type}_{kernel_name}.csv"
                    df1_filepath = os.path.join(output_dir, df1_filename)
                    df1.to_csv(df1_filepath, index=False)

                    # --- Calculate metrics for df2 ---
                    df1_test_only = df1[df1['fold'] == 1]
                    tau, _ = kendalltau(df1_test_only['y_true'], df1_test_only['y_pred'])
                    tau_sig, _ = kendalltau(df1_test_only['y_true_sigmoid'], df1_test_only['y_pred_sigmoid'])

                    metadata_list.append({
                        "id": experiment_id, "GP": model_name, "FitnessFn": fn_name,
                        "seed": SEED, "Noise_type": noise_type, "noise_level": noise_level,
                        "kernel": kernel_name, "nll": train_nll, "kendal_tau": tau,
                        "kendal_tau_sigmoid": tau_sig
                        })
                    experiment_id += 1
                    
                except Exception as e:
                    print(f"ERROR training {model_name} with {kernel_name}: {e}")
                    import traceback
                    traceback.print_exc()

    # --- 5. FINAL METADATA DATAFRAME (df2) ---
    print("\n--- Experiment Complete ---")
    df2 = pd.DataFrame(metadata_list)

    # --- Re-order and Display Results ---
    try:
        final_columns = [
            'id', 'GP', 'FitnessFn', 'seed', 'Noise_type', 'noise_level', 'kernel',
            'nll', 'kendal_tau', 'kendal_tau_sigmoid'
        ]
        df2_final = df2.reindex(columns=final_columns)

        df2_filepath = "experiment_summary.csv"
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