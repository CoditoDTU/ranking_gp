
import torch
import gpytorch
import numpy as np # For plotting
import torch.nn.functional as F





# --- Noising module --
def add_noise(y_true, noise_type='gaussian', noise_params=None):
    """
    Adds Gaussian or Heteroscedastic noise to the ground truth data.

    Args:
        y_true (torch.Tensor or np.ndarray): The ground truth values.
        noise_type (str): 'gaussian' or 'heteroscedastic'.
        noise_params (dict): Dictionary of parameters (std, base_std, amplitude_factor).

    Returns:
        The noisy data in the same format (Tensor or Array) as the input.
    """
    if noise_params is None:
        noise_params = {}

    if noise_type == 'gaussian':
        # y = y_true + N(0, std)
        std = noise_params.get('g_std', 1.0)
        std = float(std)
        #print('gstd is:', std)
        if torch.is_tensor(y_true):
            noise = torch.randn_like(y_true) * std # we get a random value with the shape of y_true and multiplied by the std
            print('y_true is a tensor')
        else:
            noise = np.random.normal(0, std, size=y_true.shape) # Normal distribution with mean 0 and std std

        return y_true + noise

    # Logic for Heteroscedastic noise
    elif noise_type == 'heteroscedastic':
        base_std = noise_params.get('h_std', 0.1)
        amplitude_factor = noise_params.get('amplitude_factor', 0.5)

        # Standard deviation scales with the magnitude of y
        # This works for both Tensors and NumPy arrays
        current_std = base_std + abs(y_true) * amplitude_factor

        if torch.is_tensor(y_true):
            noise = torch.randn_like(y_true) * current_std

        else:
            noise = np.random.normal(0, current_std)

        return y_true + noise

    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")