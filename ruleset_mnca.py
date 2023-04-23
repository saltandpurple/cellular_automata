import cupy as cp
import numpy as np

class RulesetMultipleNeighbourhoods:

    def __init__(self):
        self.neighbours = cp.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=np.uint8)

    def apply_rule(state):

