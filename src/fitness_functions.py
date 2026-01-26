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
from bayeso_benchmarks import Ackley, GramacyAndLee2012, Cosines, Levy, Bohachevsky, Sphere, Zakharov, Rosenbrock, Branin, Colville
from bayeso_benchmarks import benchmark_base


# Set a random seed for reproducibility
torch.manual_seed(42)

'''
Many Local minima Function in used from sfu.ca as the following.

As the criteria all functions are compatible to be n-dimensional


#################################################################################
Many Local Minima: Ackley, Levy

Fitness functions implementation.
Categories:
1. Many Local Minima: Ackley, Levy
2. Bowl-Shaped: Sphere, Sum of Different Powers
3. Plate-Shaped: Power Sum, Zakharov
4. Valley-shaped: Rosenbrock, Dixon-Price
5. Steep ridges: Michalewicz
'''


# ==========================================
#                FUNCTIONS
# ==========================================

# --- Many Local Minima ---

# Ackley
def fun_ackley(bx, dim_bx, a=20.0, b=0.2, c=2.0*np.pi):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    
    y = -a * np.exp(-b * np.linalg.norm(bx, ord=2, axis=0) * np.sqrt(1.0 / dim_bx)) - np.exp(1.0 / dim_bx * np.sum(np.cos(c * bx), axis=0)) + a + np.exp(1.0)
    return y
# Levy
def fun_levy(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    bw = []
    for x in bx:
        w = 1.0 + (x - 1.0) / 4.0
        bw.append(w)
    bw = np.array(bw)

    y = np.sin(np.pi * bw[0])**2

    for w in bw[:-1]:
        y += (w - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w + 1.0)**2)

    y += (bw[-1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * bw[-1])**2)
    return y

# --- Bowl-Shaped ---

# Sphere
def fun_sphere(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = 0.0
    for ind in range(0, dim_bx):
        y += bx[ind]**2
    return y

# Sum of Different Powers
def fun_sum_of_different_powers(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = 0.0
    for i in range(dim_bx):
        y += np.abs(bx[i])**(i + 2)
    return y

# --- Plate-Shaped ---

# Zakharov
def fun_zakharov(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    first_term = np.sum(bx**2)

    inner_term = 0.0
    for ind in range(1, dim_bx + 1):
        inner_term += 0.5 * ind * bx[ind - 1]

    second_term = inner_term**2
    third_term = inner_term**4

    y = first_term + second_term + third_term
    return y


# Power sum
def fun_power_sum(bx, dim_bx, b):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    
    y = 0.0
    for k in range(1, dim_bx + 1):
        inner_sum = 0.0
        for i in range(dim_bx):
            inner_sum += (i + 1)**k * bx[i]
        y += (inner_sum - b[k-1])**2
    return y



# --- Valley-shaped --

# Rosenbrock
def fun_rosenbrock(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    y = 0.0

    for ind in range(0, dim_bx - 1):
        y += 100 * (bx[ind+1] - bx[ind]**2)**2 + (bx[ind] - 1.0)**2
    return y



# Dixon-Price

def fun_dixon_price(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = (bx[0] - 1.0)**2
    for i in range(1, dim_bx):
        y += (i + 1) * (2.0 * bx[i]**2 - bx[i-1])**2
    return y


# --- Steep ridges ---

# Michalewicz
def fun_michalewicz(bx, dim_bx, m=10):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    y = 0.0
    for i in range(dim_bx):
        y += np.sin(bx[i]) * np.sin(((i + 1) * bx[i]**2) / np.pi)**(2 * m)
    return -y


# ==========================================
#                 CLASSES
# ==========================================

# --- Many Local Minima ---

class Ackley(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        bounds = np.array([
            [-32.768, 32.768],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = 0.0
        
        function = lambda bx: fun_ackley(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)

class Levy(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        bounds = np.array([
            [-10.0, 10.0],
        ])
        global_minimizers = np.array([
            [1.0],
        ])
        global_minimum = 0.0
        
        function = lambda bx: fun_levy(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)


# --- Bowl-Shaped ---

class Sphere(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        bounds = np.array([
            [-5.12, 5.12],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = 0.0
        
        function = lambda bx: fun_sphere(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)

class SumOfDifferentPowers(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        bounds = np.array([
            [-1.0, 1.0],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = 0.0
        
        function = lambda bx: fun_sum_of_different_powers(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)


# --- Plate-Shaped ---

class PowerSum(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        # Bounds are not strictly defined in some benchmarks, but usually around the solution.
        # Solution here is set to be at x = [1, 1, ..., 1]
        bounds = np.array([
            [0.0, float(dim_problem)],
        ])
        
        # Calculate b vector based on global minimizer x* = [1, 1, ..., 1]
        # b_k = sum_{i=1}^d i^k * 1
        b = []
        for k in range(1, dim_problem + 1):
            val = sum([(i + 1)**k for i in range(dim_problem)])
            b.append(val)
        b = np.array(b)

        global_minimizers = np.array([
            [1.0],
        ])
        global_minimum = 0.0
        
        function = lambda bx: fun_power_sum(bx, dim_problem, b)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)

class Zakharov(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        bounds = np.array([
            [-5.0, 10.0],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = 0.0
        
        function = lambda bx: fun_zakharov(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)


# --- Valley-shaped ---

class Rosenbrock(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))
        assert dim_problem > 1

        dim_bx = np.inf
        bounds = np.array([
            [-10.0, 10.0],
        ])
        global_minimizers = np.array([
            [1.0],
        ])
        global_minimum = 0.0
        
        function = lambda bx: fun_rosenbrock(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)

class DixonPrice(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = dim_problem
        bounds = np.array([[-10.0, 10.0]] * dim_problem)

        # Calculate exact global minimizer for Dixon-Price
        # x_1 = 1, x_i = sqrt(x_{i-1}/2)
        minimizer = np.zeros(dim_problem)
        minimizer[0] = 1.0
        for i in range(1, dim_problem):
            minimizer[i] = np.sqrt(minimizer[i-1] / 2.0)

        global_minimizers = np.array([minimizer])
        global_minimum = 0.0
        
        function = lambda bx: fun_dixon_price(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)


# --- Steep ridges ---

class Michalewicz(benchmark_base.Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = dim_problem
        bounds = np.array([[0.0, np.pi]] * dim_problem)

        # Global minimum is hard to determine for arbitrary d.
        # We use the origin (0 vector) as a reference point where f(0) = 0.
        # This satisfies the validation check: output(minimizer) == minimum.
        global_minimizers = np.zeros((1, dim_problem))
        global_minimum = 0.0
        
        function = lambda bx: fun_michalewicz(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)



# --- 2. HELPER FUNCTIONS & CLASSES ---
def fitness_function(base_fn_name: str, dimension: int = 1 ):
  
  base_fn_name = base_fn_name.lower()

  if base_fn_name == 'levy':
      benchmark = Levy(dimension)
  elif base_fn_name == 'bohachevsky':
      benchmark = Bohachevsky() # Fixed 2D in bayeso_benchmarks
  elif base_fn_name == 'ackley':
      benchmark = Ackley(dimension)
  elif base_fn_name == 'gramacy_and_lee':
      benchmark = GramacyAndLee2012() # Fixed 1D
  elif base_fn_name == 'cosines':
      benchmark = Cosines(dimension)
  elif base_fn_name == 'sphere':
      benchmark = Sphere(dimension)
  elif base_fn_name == 'sum_of_different_powers':
      benchmark = SumOfDifferentPowers(dimension)
  elif base_fn_name == 'power_sum':
      benchmark = PowerSum(dimension)
  elif base_fn_name == 'zakharov':
      benchmark = Zakharov(dimension)
  elif base_fn_name == 'rosenbrock':
      benchmark = Rosenbrock(dimension)
  elif base_fn_name == 'dixon_price':
      benchmark = DixonPrice(dimension)
  elif base_fn_name == 'michalewicz':
      benchmark = Michalewicz(dimension)
  elif base_fn_name == 'branin':
      benchmark = Branin() # Fixed 2D
  elif base_fn_name == 'colville':
      benchmark = Colville() # Fixed 4D
  else:
      raise ValueError(f"Unknown bayeso_fn_name: {base_fn_name}")

  return benchmark


# This code has been inspired in the Bayeso benchmark library from (Insert author here)