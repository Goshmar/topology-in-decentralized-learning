import numpy as np
import torch
import networkx as nx
import random

from random_isotropic_quadratics import random_quadratic_rate_precise
from topologies import AveragingScheme


class SlowChangingGraph(AveragingScheme):

    def __init__(self, num_workers, k, seed=42, init_p=0.5):
        super().__init__()
        self.num_workers = num_workers
        self.k = k
        self.init_p = init_p
        self.seed = seed
        self.iteration = 0
        random_iter = 0
        while True:
            graph = nx.erdos_renyi_graph(self.num_workers, self.init_p, seed=self.seed + random_iter)
            if nx.is_connected(graph):
                self.graph = graph
                break
            random_iter += 1

    def next_step(self):
        random_iteration = 0
        while True:
            edges_to_change = set([])
            new_graph = self.graph.copy()
            while len(edges_to_change) < self.k:
                random.seed(self.seed + random_iteration + 100 * self.iteration)
                node1 = random.randint(0, self.num_workers - 1)
                node2 = random.randint(0, node1)
                if node1 == node2 or (node1, node2) in edges_to_change:
                    random_iteration += 1
                    continue
                if self.graph.has_edge(node1, node2):
                    new_graph.remove_edge(node1, node2)
                else:
                    new_graph.add_edge(node1, node2)
                edges_to_change = edges_to_change.union(set([(node1, node2)]))
                random_iteration += 1

            if nx.is_connected(new_graph):
                self.graph = new_graph
                break
            random_iteration += 1

        self.iteration += 1

    def get_w(self):
        G = nx.to_numpy_array(self.graph)
        G = torch.tensor(G)
        num_neighbors = G.sum(1, True)
        normalization = 1 / torch.max(num_neighbors, num_neighbors.T)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G.to(torch.float32)


class SlowChangingRing(AveragingScheme):

    def __init__(self, num_workers, k, seed=42):
        super().__init__()
        self.num_workers = num_workers
        self.k = k
        self.seed = seed
        self.graph = nx.cycle_graph(num_workers)
        self.iteration = 0

    def next_step(self):
        self.graph = nx.cycle_graph(self.num_workers)
        edges_to_change = set([])
        random_iteration = 0
        while len(edges_to_change) < self.k:
            random.seed(self.seed + 100 * self.iteration + random_iteration)
            node1 = random.randint(0, self.num_workers - 1)
            node2 = random.randint(0, node1)
            if (node1 - node2) % (self.num_workers - 1) < 2 or (node1, node2) in edges_to_change:
                random_iteration += 1
                continue
            self.graph.add_edge(node1, node2)
            edges_to_change = edges_to_change.union(set([(node1, node2)]))
            random_iteration += 1
        self.iteration += 1

    def get_w(self):
        G = nx.to_numpy_array(self.graph)
        G = torch.tensor(G)
        num_neighbors = G.sum(1, True)
        normalization = 1 / torch.max(num_neighbors, num_neighbors.T)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G.to(torch.float32)


def simulate_random_walk(scheme, gamma, num_steps, num_reps):
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


# def train_random_quadratic(n_workers, zeta, scheme, lr, max_iter, tol=1e-9):
#     current_states = torch.normal(0, 1, size=(n_workers, zeta)).to(torch.float32)
#     for t in range(max_iter):
#         for worker in range(n_workers):
#             d = np.random.normal((n_workers, zeta))
#             grad = np.matmul(np.matmul(d[:, 1], d[1, :]), current_states[worker])
#             current_states[worker] -= lr * grad
#         W = scheme.get_w()
#         current_states = W @ current_states
#         scheme.next_step()

def random_quadratic_rate(scheme, zeta: float, learning_rate: float, max_iters: int = 100):
    """
    Compute the convergence rate for the random quadratic problem
    based on Section 3.1 of the paper.
    Returns a constant r for which `E x[t]^2 = (1 - r) E x[t-1]^2` or `None`.
    """
    eta = learning_rate
    r = 0
    for _ in range(max_iters):  # fixed-point iteration
        gamma = (1 - eta) ** 2 / (1 - r)
        if gamma > 1 or abs(gamma - 1) < 1e-16:
            break
        n = simulate_random_walk(scheme, gamma, num_steps=300, num_reps=30000)
        r, prev_r = 1 - (1 - eta) ** 2 - (zeta - 1) * eta ** 2 / n, r
        if abs(r - prev_r) < 1e-8:
            break
        if r >= 1 or r < 0:
            return None
    return r


if __name__ == "__main__":
    num_workers = 10
    k = 2
    lrs = torch.logspace(-3, 0, 100)
    scheme = SlowChangingRing(num_workers, k)
    conv_rates = [
        random_quadratic_rate_precise(scheme, zeta=100, learning_rate=lr.item())
        for lr in lrs
    ]
    best_conv = max([aboba for aboba in conv_rates if aboba is not None])
    print(best_conv)