import torch
import numpy as np
from matplotlib import pyplot as plt
from random_isotropic_quadratics import random_quadratic_rate
from random_isotropic_quadratics import random_quadratic_rate_precise
from topologies import scheme_for_string, Matrix
from random_walks import effective_number_of_neighbors
import networkx as nx



def spectral_gap(matrix):
    _, s, _ = torch.linalg.svd(matrix)
    abs_eigenvalues = torch.sqrt(s**2)
    return abs_eigenvalues[0] - abs_eigenvalues[1]

def main():
    torch.set_default_dtype(torch.float64)
    lrs = torch.logspace(-3, 0, 100)
    fig, ax = plt.subplots()

    opt_rates = []
    spectral_gaps = []
    enns = []

    n_workers = 64
    for p in np.linspace(start=1/2, stop=1, num=10):
        G = nx.erdos_renyi_graph(n_workers, p)
        print(nx.is_connected(G))
        G = nx.to_numpy_array(G)
        G = torch.tensor(G)

        num_neighbors = G.sum(1, True)
        normalization = 1 / torch.max(num_neighbors, num_neighbors.T)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))

        scheme = Matrix(G)
        conv_rates =  [
                random_quadratic_rate(scheme, zeta=50, learning_rate=lr.item())
                for lr in lrs
                ]
        best_conv = max([aboba for aboba in conv_rates if aboba is not None])
        spectral_gap_ = spectral_gap(scheme.w())
        eff_number = effective_number_of_neighbors(scheme, gamma=0.951)

        opt_rates.append(best_conv)
        spectral_gaps.append(spectral_gap_)
        enns.append(eff_number)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(enns, opt_rates)
    ax1.set_ylabel('Optimal convergence rate')
    ax1.set_xlabel('Effective number of neighbours')

    ax2.scatter(spectral_gaps, opt_rates)
    ax2.set_ylabel('Optimal convergence rate')
    ax2.set_xlabel('Spectral gap')

    plt.savefig('aboba')

if __name__ == "__main__":
    main()
