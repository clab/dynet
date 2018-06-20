import operator
import numpy as np


# A simple binary tree structure to calculate some statistics from leaves.
class ReduceTree(object):
    def __init__(self, size, op):
        if size & (size - 1) != 0:
            raise ValueError("size mush be a power of 2.")
        self.size = size
        self.values = np.zeros(2 * self.size)
        self.op = op

    @property
    def root(self):
        return self.values[1]

    def __setitem__(self, idx, val):
        if isinstance(idx, int):
            idx += self.size
            self.values[idx] = val
            self._percolate_up(idx, val)
        elif isinstance(idx, list):
            for i, idxx in enumerate(idx):
                idxx += self.size
                self.values[idxx] = val[i]
                self._percolate_up(idxx, val[i])
        else:
            raise RuntimeError("Not indexable type")

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.values[idx + self.size]
        elif isinstance(idx, list):
            return self.values[np.array(idx) + self.size]
        else:
            raise RuntimeError("Not indexable type")

    def _percolate_up(self, idx, val):
        idx //= 2
        while idx > 0:
            self.values[idx] = self.op(self.values[2 * idx], self.values[2 * idx + 1])
            idx //= 2


class SumTree(ReduceTree):
    def __init__(self, size):
        super().__init__(size, operator.add)

    def sample(self, value):
        idx = 1
        while idx < self.size:
            child = 2 * idx
            if value <= self.values[child]:
                idx = child
            else:
                value -= self.values[child]
                idx = child + 1

        return idx - self.size
