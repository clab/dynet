import numpy as np
from math import log, ceil

from reduce_tree import ReduceTree, SumTree


# A simple memory to store and sample experiences.
class Memory(object):
    def __init__(self, size):
        self.size = size
        self.idx = 0
        self.memory = np.zeros(size, dtype=object)

    def store(self, exp):
        self.memory[self.idx % self.size] = exp
        self.idx += 1

    def sample(self, batch_size):
        indices = np.random.choice(min(self.idx, self.size), batch_size)
        return self.memory[indices]


# Prioritized Replay: https://arxiv.org/abs/1511.05952
class PrioritizedMemory(object):
    def __init__(self, size, alpha=0.6):
        self.size = int(2 ** ceil(log(size, 2)))
        self.memory = np.zeros(self.size, dtype=object)
        self.sum_tree = SumTree(self.size)
        self.min_tree = ReduceTree(self.size, min)
        self.idx = 0
        self.max_value = 1.
        self.max_value_upper = 1000.
        self.alpha = alpha

    def store(self, exp):
        idx = self.idx % self.size
        self.memory[idx] = exp
        self.sum_tree[idx] = self.max_value
        self.min_tree[idx] = self.max_value
        self.idx += 1

    def sample(self, batch_size, beta):
        indices = []
        max_value = self.sum_tree.root
        for _ in range(batch_size):
            value = np.random.uniform(0, max_value)
            idx = self.sum_tree.sample(value)
            indices.append(idx)
        min_value = self.min_tree.root
        return indices, self.memory[indices], (self.sum_tree[indices] / (min_value + 1e-4)) ** (-beta)

    def update(self, indices, values):
        values = np.array(values)
        values_modified = values ** self.alpha
        self.sum_tree[indices] = values_modified
        self.min_tree[indices] = values_modified
        self.max_value = max(self.max_value, np.max(values))
        self.max_value = min(self.max_value, self.max_value_upper)

    def is_full(self):
        return self.idx >= self.size
