import numpy as np
import torch
from typing import List

def simulate_random_walk(scheme, gamma, num_steps=300, num_reps=3000):
    x = torch.zeros([scheme.num_workers, num_reps], dtype=torch.float32)
    y = torch.zeros([scheme.num_workers, num_reps], dtype=torch.float32)
    
    for t in range(num_steps):
        W = scheme.get_w()
        x = torch.matmul(W, torch.sqrt(gamma) * x + torch.randn_like(x))
        
        scheme.next_step()
        y = torch.sqrt(gamma) * y + torch.randn_like(y)

    x_var = torch.var(x, dim=1).sum()
    y_var = torch.var(y, dim=1).sum()

    return y_var / x_var

def spectral_gap(matrix):
    _, s, _ = torch.linalg.svd(matrix)
    abs_eigenvalues = torch.sqrt(s ** 2)
    return abs_eigenvalues[0] - abs_eigenvalues[1]
