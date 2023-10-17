import numpy as np
import torch
import networkx as nx
import random
from topologies import AveragingScheme

class SlowChangingGraph:

    def __init__(self, num_workers, k, seed=42, init_p=0.5):
        """
        Initialize a SlowChangingGraph instance.
        """
        self.num_workers = num_workers
        self.k = k
        self.init_p = init_p
        self.seed = seed
        self.iteration = 0
        self._initialize_graph()

    def _initialize_graph(self):
        # Initialize a connected graph
        random_iter = 0
        while True:
            graph = nx.erdos_renyi_graph(self.num_workers, self.init_p, seed=self.seed + random_iter)
            if nx.is_connected(graph):
                self.graph = graph
                break
            random_iter += 1

    def next_step(self):
        """
        Move to the next step by changing edges in the graph.
        """
        random_iteration = 0
        while True:
            edges_to_change = set()
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
                edges_to_change.add((node1, node2))
                random_iteration += 1

            if nx.is_connected(new_graph):
                self.graph = new_graph
                break
            random_iteration += 1

        self.iteration += 1

    def get_w(self):
        """
        Get the adjacency matrix for the current graph.
        """
        G = nx.to_numpy_array(self.graph)
        G = torch.tensor(G, dtype=torch.float32)
        num_neighbors = G.sum(1, keepdim=True)
        normalization = 1 / (torch.max(num_neighbors, num_neighbors.T) + 1)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G


class SlowChangingRing:

    def __init__(self, num_workers, k, seed=42):
        """
        Initialize a SlowChangingRing instance.
        """
        self.num_workers = num_workers
        self.k = k
        self.seed = seed

    def w(self, t=0, params=None):
        """
        Generate the adjacency matrix for the ring graph.
        """
        graph = nx.cycle_graph(self.num_workers)
        edges_to_change = set()
        random_iteration = 0
        while len(edges_to_change) < self.k:
            random.seed(self.seed + 100 * t + random_iteration)
            node1 = random.randint(0, self.num_workers - 1)
            node2 = random.randint(0, node1)
            if (node1 - node2) % (self.num_workers - 1) < 2 or (node1, node2) in edges_to_change:
                random_iteration += 1
                continue
            graph.add_edge(node1, node2)
            edges_to_change.add((node1, node2))
            random_iteration += 1

        G = nx.to_numpy_array(graph)
        G = torch.tensor(G, dtype=torch.float32)
        num_neighbors = G.sum(1, keepdim=True)
        normalization = 1 / (torch.max(num_neighbors, num_neighbors.T) + 1)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G


class SlowChangingStar:

    def __init__(self, num_workers, k, seed=42):
        """
        Initialize a SlowChangingStar instance.
        """
        self.num_workers = num_workers
        self.k = k
        self.seed = seed

    def w(self, t=0, params=None):
        """
        Generate the adjacency matrix for the star graph.
        """
        graph = nx.star_graph(self.num_workers)
        edges_to_change = set()
        random_iteration = 0
        while len(edges_to_change) < self.k:
            random.seed(self.seed + 100 * t + random_iteration)
            node1 = random.randint(0, self.num_workers - 1)
            node2 = random.randint(0, node1)
            if node2 == node1 or node2 == 0 or (node1, node2) in edges_to_change:
                random_iteration += 1
                continue
            graph.add_edge(node1, node2)
            edges_to_change.add((node1, node2))
            random_iteration += 1

        G = nx.to_numpy_array(graph)
        G = torch.tensor(G, dtype=torch.float32)
        num_neighbors = G.sum(1, keepdim=True)
        normalization = 1 / (torch.max(num_neighbors, num_neighbors.T) + 1)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G

class TimeVaringErdos(AveragingScheme):

    def __init__(self, n, p, seed=None):
        """
        Initialize a TimeVaryingErdos instance.
        """
        super().__init__()
        self.n = n
        self.p = p
        self.seed = seed
        self.period = 100

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
        num_neighbors = G.sum(1, keepdim=True)
        normalization = 1 / (torch.max(num_neighbors, num_neighbors.T) + 1)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G


class Erdos(AveragingScheme):

    def __init__(self, n, p):
        """
        Initialize an Erdos instance.
        """
        super().__init__()
        self.n = n
        self.p = p
        self.G = self._generate_graph()

    def _generate_graph(self):
        G = nx.erdos_renyi_graph(self.n, self.p)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(self.n, self.p)
        G = nx.to_numpy_array(G)
        G = torch.tensor(G, dtype=torch.float32)
        num_neighbors = G.sum(1, keepdim=True)
        normalization = 1 / (torch.max(num_neighbors, num_neighbors.T) + 1)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G

    def w(self, t=0, params=None):
        return self.G


class Ring:

    def __init__(self, n):
        """
        Initialize a Ring instance.
        """
        self.n = n
        self.G = self._generate_graph()

    def _generate_graph(self):
        G = nx.cycle_graph(self.n)
        G = nx.to_numpy_array(G)
        G = torch.tensor(G, dtype=torch.float32)
        num_neighbors = G.sum(1, keepdim=True)
        normalization = 1 / (torch.max(num_neighbors, num_neighbors.T) + 1)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G

    def w(self, t=0, params=None):
        return self.G


class Star:

    def __init__(self, n):
        """
        Initialize a Star instance.
        """
        self.n = n
        self.G = self._generate_graph()

    def _generate_graph(self):
        G = nx.star_graph(self.n)
        G = nx.to_numpy_array(G)
        G = torch.tensor(G, dtype=torch.float32)
        num_neighbors = G.sum(1, keepdim=True)
        normalization = 1 / (torch.max(num_neighbors, num_neighbors.T) + 1)
        G = normalization * G
        G += torch.diag(1 - G.sum(1))
        return G

    def w(self, t=0, params=None):
        return self.G

class CycleChangingGraph:

    def __init__(self, n, topologies):
        """
        Initialize a CycleChangingGraph instance.
        """
        self.n = n
        self.topologies = topologies

    def w(self, t=0, params=None):
        graph = t % len(self.topologies)
        step = t // len(self.topologies)
        return self.topologies[graph].w(step)
