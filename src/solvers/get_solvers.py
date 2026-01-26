from typing import Iterable
import torch


# Helper function to get optimizer
def get_optimizer(optimizer_name: str, parameters: Iterable, lr: float) -> torch.optim.Optimizer:
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