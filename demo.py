import torch
import numpy as np
from matplotlib import pyplot as plt
from random_isotropic_quadratics import random_quadratic_rate
from random_isotropic_quadratics import random_quadratic_rate_precise
from topologies import scheme_for_string, Matrix, AveragingScheme, lru_cache
from random_walks import effective_number_of_neighbors
import networkx as nx
import seaborn as sns
import pandas as pd

class TimeVaringErdos(AveragingScheme):
    def __init__(self, n, p):
        super().__init__()
        self.n = n
        self.p = p

    @lru_cache(maxsize=10)
    def w(self, t=0, params=None):
        return nx.erdos_renyi_graph(self.n, self.p)


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
    probs = []

    time_variance_scheme = TimeVaringErdos(64, 0.5)
    
    n_workers = 64
    # for p in np.linspace(start=0, stop=1, num=100):
    #     G = nx.erdos_renyi_graph(n_workers, p)
    for p in np.linspace(start=0, stop=1, num=20):
        for _ in range(20):
            G = nx.erdos_renyi_graph(n_workers, p)
            # G = nx.barabasi_albert_graph(n_workers, m)
            # G = nx.random_regular_graph(d=m , n=n_workers)
            if not nx.is_connected(G):
                continue
            G = nx.to_numpy_array(G)
            G = torch.tensor(G)

            num_neighbors = G.sum(1, True)
            normalization = 1 / torch.max(num_neighbors, num_neighbors.T)
            G = normalization * G
            G += torch.diag(1 - G.sum(1))

            scheme = Matrix(G)
            conv_rates =  [
                    random_quadratic_rate(scheme, zeta=100, learning_rate=lr.item())
                    for lr in lrs
                    ]
            best_conv = max([aboba for aboba in conv_rates if aboba is not None])
            spectral_gap_ = spectral_gap(scheme.w())

            eff_number = None
            for gamma in np.linspace(0.01, 1, num=100):
                new_eff_number = effective_number_of_neighbors(scheme, gamma=gamma)
                if 2 * new_eff_number**2 / (1 - gamma * (1 - spectral_gap_)) <= 32 * 50:
                    eff_number = new_eff_number
                else:
                    break
            opt_rates.append(np.log(best_conv))
            spectral_gaps.append(spectral_gap_)
            enns.append(eff_number)
            # probs.append(p)
            probs.append(p)
            # probs.append(d)
    stacked_np = np.zeros((len(opt_rates), 4))
    stacked_np[:, 0] = np.array(opt_rates)
    stacked_np[:, 1] = np.array(spectral_gaps)
    stacked_np[:, 2] = np.array(enns)
    stacked_np[:, 3] = np.array(probs)

    param_name = 'm'
    stacked_df = pd.DataFrame(stacked_np,
                              columns=['opt_rates', 'spectral_gaps', 'enns', param_name])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.scatterplot(ax=ax1, data=stacked_df, x='enns', y='opt_rates', hue=param_name)
    ax1.set_ylabel('Optimal convergence rate')
    ax1.set_xlabel('Effective number of neighbours')

    sns.scatterplot(ax=ax2, data=stacked_df, x='spectral_gaps', y='opt_rates', hue=param_name)
    ax2.set_ylabel('Optimal convergence rate')
    ax2.set_xlabel('Spectral gap')

    plt.savefig('aboba_1')

if __name__ == "__main__":
    main()
