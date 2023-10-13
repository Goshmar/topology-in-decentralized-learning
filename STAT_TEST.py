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
from demo import simulate_random_walk
from tqdm import trange, tqdm
import scipy.stats as sps


class TimeVaringErdos(AveragingScheme):
    def __init__(self, n, p, seed=None):
        self.period = 100
        super().__init__()
        self.n = n
        self.p = p
        self.seed = seed
    @lru_cache(maxsize=10)
    def w(self, t=0, params=None):
        if self.seed is not None:
            G = nx.erdos_renyi_graph(self.n, self.p, seed=self.seed + t)
        else:
            G = nx.erdos_renyi_graph(self.n, self.p)

        while not nx.is_connected(G):
            if self.seed is not None:
                G = nx.erdos_renyi_graph(self.n, self.p, seed=self.seed + t)
            else:
                G = nx.erdos_renyi_graph(self.n, self.p)

        G = nx.to_numpy_array(G)
        G = torch.tensor(G, dtype=torch.float32)
        num_neighbors = G.sum(1, True)
        normalization = 1 / torch.max(num_neighbors, num_neighbors.T)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G

class Erdos(AveragingScheme):
    def __init__(self, n, p):
        self.n = n
        self.p = p
        G = nx.erdos_renyi_graph(self.n, self.p)
        G = nx.to_numpy_array(G)
        G = torch.tensor(G, dtype=torch.float32)
        num_neighbors = G.sum(1, True)
        normalization = 1 / torch.max(num_neighbors, num_neighbors.T)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        self.G = G

    def w(self, t=0,  params=None):
        return self.G

if __name__ == '__main__':
    n_varying = 100

    n_const = 200

    samples_const_for_p = []
    samples_var_for_p = []

    for p in tqdm(np.linspace(1 / 10, 9 / 10, num=20)):
        samples_varying = []
        for _ in trange(n_varying):
            scheme = TimeVaringErdos(n=64, p=p)
            samples_varying.append(simulate_random_walk(scheme,
                                                    gamma=0.951,
                                                    num_steps=1000,
                                                    num_workers=64,
                                                    num_reps=1000))
        samples_var_for_p.append(samples_varying)

        samples_const = []
        for _ in trange(n_const):
            scheme = Erdos(n=64, p=p)
            samples_const.append(simulate_random_walk(scheme,
                                                gamma=0.951,
                                                num_steps=1000,
                                                num_workers=64,
                                                num_reps=1000))
        samples_const_for_p.append(samples_const)

    samples_var_for_p = torch.tensor(samples_var_for_p)
    samples_const_for_p = torch.tensor(samples_const_for_p)

    torch.save(samples_var_for_p, 'samples_var_for_p.pt')
    torch.save(samples_const_for_p, 'samples_const_for_p.pt')

    print(samples_var_for_p.shape, samples_const_for_p.shape)