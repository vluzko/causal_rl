import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gym import Env
from scipy.spatial import distance
from typing import Optional, Tuple, Any

from causal_rl.environments import CausalEnv


class HardSpheres(CausalEnv):
    """A simulation of a hard-sphere gas model.

    Adapted from MagNet

    Attributes:
        num_obj (int):          Number of balls in the simulation.
        obj_dim (int):          The dimension of the balls. Will always be 2 * dimension_of_space
        masses (np.ndarray):    The masses of the balls.
        radii (np.ndarray):     The radii of the balls.
        space (pymunk.Space):   The actual simulation space.
    """

    cls_name = "bouncing_balls"

    def __init__(
        self,
        num_obj: int = 5,
        mass: float = 10.0,
        radius: float = 10.0,
        width: float = 150.0,
    ):
        self.num_obj = num_obj
        self.obj_dim = 4
        self.mass = mass
        self.radius = radius
        self.masses = mass * np.ones(num_obj)
        self.radii = radius * np.ones(num_obj)
        self.width = width
        self.location_indices = (0, 1)

    @property
    def name(self) -> str:
        return "bouncing_balls_{}_{}_{}_{}".format(
            self.num_obj, self.mass, self.radius, self.width
        )

    def reset(self):
        import pymunk

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.objects = []

        pos_scale = self.width - self.radius * 2

        x_pos = np.random.rand(self.num_obj, 1) * pos_scale + self.radius
        y_pos = np.random.rand(self.num_obj, 1) * pos_scale + self.radius
        x_vel = np.random.rand(self.num_obj, 1) * 400 - 200
        y_vel = np.random.rand(self.num_obj, 1) * 400 - 200

        for i in range(self.num_obj):
            mass = self.masses[i]
            radius = self.radii[i]
            moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            body = pymunk.Body(mass, moment)
            body.position = (x_pos[i], y_pos[i])
            body.velocity = (x_vel[i], y_vel[i])
            shape = pymunk.Circle(body, radius, (0, 0))
            shape.elasticity = 1.0
            self.space.add(body, shape)
            self.objects.append(body)

        static_body = self.space.static_body
        static_lines = [
            pymunk.Segment(static_body, (0.0, 0.0), (0.0, self.width), 0),
            pymunk.Segment(static_body, (0.0, 0.0), (self.width, 0.0), 0),
            pymunk.Segment(static_body, (self.width, 0.0), (self.width, self.width), 0),
            pymunk.Segment(static_body, (0.0, self.width), (self.width, self.width), 0),
        ]

        for line in static_lines:
            line.elasticity = 1.0
        self.space.add(static_lines)

        return self.get_state(), 0, False, None

    def get_state(self) -> np.ndarray:
        """Get the current state.

        Returns:
            A tensor representing the state. Each row is a single ball, columns are [*position, *velocity]
        """
        state = np.zeros((self.num_obj, 4))
        for i in range(self.num_obj):
            state[i, :2] = np.array(
                [self.objects[i].position[0], self.objects[i].position[1]]
            )
            state[i, 2:] = np.array(
                [self.objects[i].velocity[0], self.objects[i].velocity[1]]
            )

        return state

    def step(self, dt=0.01) -> Tuple[np.ndarray, float, bool, Any]:
        self.space.step(dt)
        return self.get_state(), 0, False, None

    # TODO: Rename `epochs` to `steps`
    def generate_data(
        self, epochs: int = 10000, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a full run of the environment."""
        states = np.zeros((epochs, self.num_obj, 4))
        rewards = np.zeros((epochs, 1))
        self.reset()

        for t in range(epochs):
            states[t] = self.get_state()
            if t > 0:
                states[t, :, 2:] = (states[t, :, :2] - states[t - 1, :, :2]) / dt
            self.step(dt=dt)

        return states, rewards

    def visualize(self, state: np.ndarray, save_path: Optional[str] = None):
        """Visualize a single state.

        Args:
            state: State at a single time step.
            save_path: Where to save the snapshot
        """
        colors = ["b", "g", "r", "c", "m", "y", "k"]
        pos = state[:, :2]
        momenta = state[:, 2:]
        fig, ax = plt.subplots(figsize=(6, 6))
        box = plt.Rectangle(
            (0, 0), self.width, self.width, linewidth=5, edgecolor="k", facecolor="none"
        )
        ax.add_patch(box)
        for i in range(len(pos)):
            circle = plt.Circle(
                (pos[i, 0], pos[i, 1]), radius=self.radii[i], edgecolor="b"
            )
            label = ax.annotate(
                "{}".format(i), xy=(pos[i, 0], pos[i, 1]), fontsize=8, ha="center"
            )
            # Plot the momentum
            plt.arrow(pos[i, 0], pos[i, 1], momenta[i, 0], momenta[i, 1])
            ax.add_patch(circle)
        plt.axis([0, 128, 0, 128])
        plt.axis("off")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def detect_collisions(self, trajectories: np.ndarray) -> np.ndarray:
        n = trajectories.shape[0]
        k = self.num_obj
        min_dist = self.radii.reshape(k, 1) + self.radii.reshape(1, k)
        np.fill_diagonal(min_dist, 0)

        collisions = np.zeros((n, k, k))

        for i in range(1, n):
            # The (x, y) coordinates of all balls at t=i
            locs = trajectories[i, :, :2]

            distances = distance.squareform(distance.pdist(locs))
            collided = np.nonzero(distances < min_dist)

            collisions[i - 1][collided] = 1

            collisions[i][collided] = 1

        return collisions

    def wall_collisions(self, states: np.ndarray) -> np.ndarray:

        min_coord = 0 + self.radius
        max_coord = self.width - self.radius

        # Just the position coordinates
        locs = states[:, :, :2]

        has_collision = (locs < min_coord) | (locs > max_coord)

        return has_collision
