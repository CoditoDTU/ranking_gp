# Cell 1: Imports
import math
import torch
import gpytorch
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import torch.nn.functional as F
import botorch
import numpy as np
import pandas as pd
from bayeso_benchmarks import Ackley, GramacyAndLee2012, Cosines, Levy, Bohachevsky
from bayeso_benchmarks import benchmark_base


# Set a random seed for reproducibility
torch.manual_seed(42)

'''
Many Local minima Function in used from sfu.ca as the following.

As the criteria all functions are compatible to be n-dimensional


#################################################################################
Many Local Minima: Ackley, Levy

Bowl-Shaped: sphere, sum of different powers function

Plate-Shaped: Power sum function, ZAKHAROV FUNCTION

Valley-shaped: ROSENBROCK FUNCTION, DIXON-PRICE FUNCTION

Steep ridges: MICHALEWICZ FUNCTION

Other: Branin, Colville
'''



# --- 2. HELPER FUNCTIONS & CLASSES ---
def fitness_function(base_fn_name: str, dimension: int = 1 ):
  
  if base_fn_name == 'levy':
      benchmark = Levy(dimension)
  elif base_fn_name == 'bohachevsky':
      benchmark = Bohachevsky(dimension)
  elif base_fn_name == 'ackley':
      benchmark = Ackley(dimension)
  elif base_fn_name == 'gramacy_and_lee':
      benchmark = GramacyAndLee2012(dimension)
  elif base_fn_name == 'cosines':
      benchmark = Cosines(dimension)
  else:
      raise ValueError(f"Unknown bayeso_fn_name: {base_fn_name}")

  return benchmark
