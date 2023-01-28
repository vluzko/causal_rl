import numpy as np
from scipy.spatial import distance

from causal_rl.environments import CausalEnv


class Torus(CausalEnv):
    """A causal environment that produces dense causal graphs"""

    def __init__(
        self,
        num_obj: int,
        min_radius: float = 0.1,
        max_radius: float = 0.15,
        dt: float = 0.01,
    ):
        self.num_obj = num_obj
        self.obj_dim = 4
        self.state = np.zeros((self.num_obj, self.obj_dim))
        self.dt = dt
        self.min_radius = min_radius
        self.max_radius = max_radius

    def step(self):
        raise NotImplementedError

    def generate_data(self, length: int):
        # Toroidal space
        self.state[:, :2] = np.random.uniform(0, 1, (self.num_obj, 2))
        self.state[:, 2:] = np.random.uniform(-1, 1, (self.num_obj, 2))
        states = []
        graphs = []
        for i in range(length):
            self.state[:, :2] += self.state[:, 2:] * self.dt
            self.state[:, :2] %= 1

            # apply forces
            distances = distance.squareform(distance.pdist(self.state[:, :2]))
            in_range = (self.min_radius < distances) & (distances < self.max_radius)
            contributions = in_range @ self.state[:, 2:] * 0.1

            self.state[:, 2:] += contributions

            states.append(self.state)
            graphs.append(in_range)

        return np.array(states), np.array(graphs)


class RotatingVectors(CausalEnv):
    """A causal environment that produces dense causal graphs"""

    def __init__(self, num_obj: int):
        self.num_obj = num_obj

    def step(self):
        raise NotImplementedError

    def generate_data(self, length: int):
        # For each vector, compute vectors that are within theta radians of it
        # Update each vector based on the nearby vectors

        raise NotImplementedError
