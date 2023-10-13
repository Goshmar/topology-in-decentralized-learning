import numpy as np
import torch


def simulate_random_walk(scheme, gamma, num_steps=300, num_reps=3000):
    x = torch.zeros([scheme.num_workers, num_reps]).to(torch.float32)
    for t in range(num_steps):
        AAA = np.sqrt(gamma) * x + torch.randn_like(x)
        W = scheme.get_w()
        x = W @ AAA
        scheme.next_step()
    x = torch.var(x, 1).sum()

    y = torch.zeros([scheme.num_workers, num_reps]).to(torch.float32)
    for t in range(num_steps):
        y = np.sqrt(gamma) * y + torch.randn_like(y)
    y = torch.var(y, 1).sum()

    return y / x


def spectral_gap(matrix):
    _, s, _ = torch.linalg.svd(matrix)
    abs_eigenvalues = torch.sqrt(s ** 2)
    return abs_eigenvalues[0] - abs_eigenvalues[1]
