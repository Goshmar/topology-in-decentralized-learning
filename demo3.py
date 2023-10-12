from functools import lru_cache
import numpy as np
import torch
import networkx as nx
import random
import matplotlib.pyplot as plt
from topologies import scheme_for_string, Matrix, AveragingScheme, lru_cache
from random_walks import effective_number_of_neighbors
from topologies import AveragingScheme


class SlowChangingGraph(AveragingScheme):

    def __init__(self, num_workers, k, seed=42, init_p=0.5):
        super().__init__()
        self.num_workers = num_workers
        self.k = k
        self.init_p = init_p
        self.seed = seed

    @lru_cache(maxsize=10)
    def w(self, t=0, params=None):

        while True:
            graph = nx.erdos_renyi_graph(self.num_workers, self.init_p, seed=self.seed)
            if nx.is_connected(graph):
                self.graph = graph
                break

        for time in range(t):
            random_iteration = 0
            while True:
                random.seed(self.seed + random_iteration + time)
                new_graph = self.graph.copy()
                for _ in range(self.k):
                    node1 = random.randint(0, self.num_workers-1)
                    node2 = random.randint(0, node1)
                    if node1 == node2:
                        continue

                    if new_graph.has_edge(node1, node2):
                        new_graph.remove_edge(node1, node2)
                    else:
                        new_graph.add_edge(node1, node2)

                if nx.is_connected(new_graph):
                    self.graph = new_graph
                    break

                random_iteration += 1

        G = nx.to_numpy_array(self.graph)
        G = torch.tensor(G)
        num_neighbors = G.sum(1, True)
        normalization = 1 / torch.max(num_neighbors, num_neighbors.T)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G
        # return self.make_gossip_W()

    # def make_gossip_W(self):
    #
    #     W = torch.zeros((self.num_workers, self.num_workers))
    #     for i in range(self.num_workers):
    #         for j in range(self.num_workers):
    #             if i < j and j in set(self.graph.neighbors(i)):
    #                 max_degree = max(self.graph.degree(i), self.graph.degree(j))
    #                 W[i][j] = 1 / (max_degree + 1)
    #                 W[j][i] = 1 / (max_degree + 1)
    #
    #     for i in range(self.num_workers):
    #         W[i, i] = 1 - W[i, :].sum()
    #
    #     return W


def simulate_random_walk(scheme, gamma, num_steps, num_reps, num_workers):

    x = torch.zeros([num_workers, num_reps])
    for t in range(num_steps):
        AAA = (np.sqrt(gamma) * x + torch.randn_like(x)).to(torch.float64)
        x = scheme.w(t) @ AAA
    x = torch.var(x, 1).sum()

    y = torch.zeros([num_workers, num_reps])
    for t in range(num_steps):
        y = (np.sqrt(gamma) * y + torch.randn_like(y)).to(torch.float64)
    y = torch.var(y, 1).sum()

    return y/x


if __name__ == "__main__":
    num_workers = 15
    for k in range(1, 56*2, 10):
        scheme = SlowChangingGraph(num_workers, k)
        a = simulate_random_walk(scheme, gamma=0.5, num_steps=5000, num_reps=5000, num_workers=num_workers)
        print(a)