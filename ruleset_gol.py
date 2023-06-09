import cupy as cp
import numpy as np
from cupyx.scipy import signal
from ruleset_interface import RulesetInterface


class RulesetGameOfLife(RulesetInterface):

    def __init__(self):
        super().__init__()
        self.neighbours = cp.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=np.int8)

    """
    GOL rules:
    A cell survives, if it has 2 or 3 neighbours.
    A cell is born, if it has exactly 3 neighbours.
    Otherwise, the cell dies/remains dead.
    """
    def calculate_next_state(self, state):
        num_neighbors = signal.convolve2d(state, self.neighbours, mode='same', boundary='wrap')
        return cp.logical_or(num_neighbors == 3, cp.logical_and(num_neighbors == 2, state)).astype(cp.int8)
