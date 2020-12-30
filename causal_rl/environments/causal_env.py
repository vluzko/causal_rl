import numpy as np

from gym import Env
from typing import Tuple

class CausalEnv(Env):
    """A causal environment.

    Attributes:
        num_obj (int): The number of objects in the environment.
        obj_dim (int): The dimension of each object
    """

    def generate_data(self, steps: int=10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a full run of the environment.

        Args:
            steps: The number of steps to run for.

        Returns:
            The array of states, and the array of rewards.
        """
        raise NotImplementedError

    def detect_collisions(self, trajectories: np.ndarray) -> np.ndarray:
        """Detect collisions for a sequence of states.

        Args:
            trajectories: The sequence of states. Shape is (n, num_obj, obj_dim)
        Returns:
            The collisions array. Shape is (n, num_obj, num_obj)
        """
        raise NotImplementedError


