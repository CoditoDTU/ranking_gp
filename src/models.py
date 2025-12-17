import torch
import gpytorch
import numpy as np # For plotting
import torch.nn.functional as F
from botorch.models import PairwiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
import sklearn
from gpytorch.kernels import (
    ScaleKernel, MaternKernel, RBFKernel, 
    PeriodicKernel, LinearKernel
)
from gpytorch.priors import GammaPrior
from gpytorch.constraints import Interval, GreaterThan




# --- Flexible ExactGP Class ---

class FlexibleExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, covar_module):
        super().__init__(train_inputs, train_targets, likelihood)
        self.covar_module = covar_module
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


'''
KERNELS ##################################################

'''

def build_kernel(kernel_name : str,
                 input_dim : int,
                 fixed_lengthscale: bool = True,
                 lengthscale: float = 1):
  """
  Builds a GPyTorch covariance module (kernel) based on a name.
  Can fix the lengthscale to a specific value.
  """

  if kernel_name == 'squared_exponential':
        base_kernel = RBFKernel()
  elif kernel_name == 'matern_5_2':
      base_kernel = MaternKernel(nu=2.5)

  elif kernel_name == 'matern_3_2':
        base_kernel = MaternKernel(nu=1.5)

  elif kernel_name == 'exponential':
        base_kernel = MaternKernel(nu=0.5)

  elif kernel_name == 'periodic':
        base_kernel = PeriodicKernel()

  elif kernel_name == 'linear':
        base_kernel = LinearKernel()
  else:
        raise ValueError(f"Unknown kernel name: '{kernel_name}'.")

  if fixed_lengthscale:
      base_kernel.lengthscale = torch.tensor([[lengthscale]], dtype = torch.float64)

        # Linear kernel has no lengthscale, so we can return early

  return ScaleKernel(base_kernel)



'''
LOSS: the Loss is currently an Import 
'''