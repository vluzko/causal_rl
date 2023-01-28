import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from gym import Env
from scipy.spatial import distance
from typing import Optional, Tuple, Any

from causal_rl.environments import CausalEnv


class Mujoco(CausalEnv):
    """A simulation of balls bouncing with gravity and elastic collisions.

    Attributes:
        num_obj (int):          Number of balls in the simulation.
        obj_dim (int):          The dimension of the balls. Will always be 2 * dimension_of_space
        masses (np.ndarray):    The masses of the balls.
        radii (np.ndarray):     The radii of the balls.
        space (pymunk.Space):   The actual simulation space.
    """

    cls_name = "ant"

    def __init__(self, env_name: str = "Hopper-v2"):
        super().__init__()
        self.obj_dim = 1
        self.env_name = env_name
        self.underlying = gym.make(self.env_name)
        self.obs_size = self.underlying.observation_space.sample().shape[0]
        self.act_size = self.underlying.action_space.sample().shape[0]
        self.num_obj = self.obs_size

    @property
    def name(self) -> str:
        return "{}_{}".format(self.cls_name, self.env_name)

    def generate_data(
        self, length: int = 1000, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        states = np.zeros((length, self.num_obj, self.obj_dim))
        rewards = np.zeros((length, 1))
        self.underlying.reset()

        for t in range(length):
            # Step with a random agent
            action = self.underlying.action_space.sample()
            state, reward, done, _ = self.underlying.step(action)
            # states[t, :self.obs_size] = state.reshape(self.obs_size, self.obj_dim)
            # states[t, self.obs_size:] = action.reshape(self.act_size, self.obj_dim)
            states[t] = state.reshape(self.obs_size, self.obj_dim)
            rewards[t] = reward

            # if done:
            #     self.underlying.reset()

        return states, rewards

    def visualize(self, state: np.ndarray, save_path: Optional[str] = None):
        """Visualize a single state.

        Args:
            state: The full
            save_path: Path to save the image to.
        """
        raise NotImplementedError

    def detect_collisions(self, trajectories: np.ndarray) -> np.ndarray:
        """For now we just return an empty collision graph."""
        n = trajectories.shape[0]
        k = self.num_obj

        collisions = np.zeros((n, k, k))

        return collisions
