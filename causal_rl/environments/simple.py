"""Purely mathematical environments with no physical representation."""
import numpy as np

from scipy.spatial import distance
from typing import Tuple

from causal_rl.environments.causal_env import CausalEnv


class SimpleEnv(CausalEnv):

    def connections(self, state: np.ndarray):
        """Choose connections for a given state."""
        raise NotImplementedError


class Sinusoidal(SimpleEnv):

    def __init__(self, num_obj: int, amplitude: float=1.0):
        super().__init__()
        self.num_obj = num_obj
        self.amplitude = amplitude

    def reset(self):
        self.state = np.random(self.num_obj)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        phases = np.sin(self.state)
        distances = distance.squareform(distance.pdist(phases))
        raise NotImplementedError

    def generate_data(self, steps: int=10000) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def generate_causal_graphs(self, trajectories: np.ndarray) -> np.ndarray:
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

    def generate_data(self, steps: int=10000) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def generate_causal_graphs(self, trajectories: np.ndarray) -> np.ndarray:
        raise NotImplementedError