"""Purely mathematical environments with no physical representation."""
import numpy as np

from typing import Tuple

from causal_rl.environments import CausalEnv


class SimpleEnv(CausalEnv):

    def connections(self, state: np.ndarray):
        """Choose connections for a given state."""
        raise NotImplementedError


class LinearGaussian(SimpleEnv):
    """An environment where each state is the unweighted sum of its parents.

    """

    def __init__(self, num_obj: int):
        self.num_obj = num_obj

    def step(self):
        # Choose connections
        # Step state
        raise NotImplementedError