import torch
import gpytorch


# --- Fixed Sigmoid Function ---
def apply_sigmoid(f: torch.Tensor, k: float, x0: float) -> torch.Tensor:
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.double)
    f = f.to(torch.double)
    return 1.0 / (1.0 + torch.exp(-k * (f - x0)))


# --- Comparisons function --- 
def get_comparisons(y_dp, epsilon=0.01):

    sorted_y, sort_idx = torch.sort(y_dp.flatten(), descending=True)
    utility_matrix = sorted_y.unsqueeze(1) - sorted_y.unsqueeze(0) #get's us the NxN matrix

    triu_mask = torch.triu(torch.ones_like(utility_matrix), diagonal=1).bool() #only true for upper triangular.
    valid_mask = triu_mask & (utility_matrix > epsilon) #only true for utility > epsilon

    i_sorted, j_sorted = torch.where(valid_mask)
    valid_utilities = utility_matrix[i_sorted, j_sorted]
    utilities_sorted, util_sort_idx = torch.sort(valid_utilities, descending=True)

    i_sorted = i_sorted[util_sort_idx]
    j_sorted = j_sorted[util_sort_idx]

    original_i = sort_idx[i_sorted]
    original_j = sort_idx[j_sorted]

    comparisons = torch.stack([original_i, original_j], dim=1)

    return comparisons