import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gym import Env
from scipy.spatial import distance
from typing import Optional, Tuple, Any

from causal_rl.environments import CausalEnv


class MultiTyped(CausalEnv):
    """A simulation of balls bouncing with gravity and elastic collisions.

    Attributes:
        num_obj (int):          Number of balls in the simulation.
        obj_dim (int):          The dimension of the balls. Will always be 2 * dimension_of_space
        masses (np.ndarray):    The masses of the balls.
        radii (np.ndarray):     The radii of the balls.
        space (pymunk.Space):   The actual simulation space.
    """

    cls_name = "multi_typed"

    def __init__(
        self, num_obj: int = 5, mass: float = 10, radii: float = 10, width: float = 400
    ):
        self.num_obj = 2 * num_obj
        self.obj_dim = 4
        self.mass = mass
        self.radius = radii
        self.masses = mass * np.ones(self.num_obj)
        self.radii = radii * np.ones(self.num_obj)
        self.width = width
        self.location_indices = (0, 1)

    @property
    def name(self) -> str:
        return "{}_{}_{}_{}".format(self.cls_name, self.mass, self.radius, self.width)

    def reset(self):
        import pymunk

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.objects = []

        x_pos = np.random.rand(self.num_obj, 1) * (self.width - 40) + 20
        y_pos = np.random.rand(self.num_obj, 1) * (self.width - 40) + 20
        x_vel = np.random.rand(self.num_obj, 1) * 300 - 150
        y_vel = np.random.rand(self.num_obj, 1) * 300 - 150

        # Create circles
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

        # Create squares
        for i in range(self.num_obj):
            mass = self.masses[i] * 6
            radius = self.radii[i] * 1.2
            size = (radius, radius)
            moment = pymunk.moment_for_box(mass, size)
            body = pymunk.Body(mass, moment)
            body.position = (x_pos[i], y_pos[i])
            body.velocity = (x_vel[i], y_vel[i])
            shape = pymunk.Poly.create_box(body, size)
            shape.elasticity = 1.0
            self.space.add(body, shape)
            self.objects.append(body)

        static_lines = [
            pymunk.Segment(self.space.static_body, (0.0, 0.0), (0.0, self.width), 0),
            pymunk.Segment(self.space.static_body, (0.0, 0.0), (self.width, 0.0), 0),
            pymunk.Segment(
                self.space.static_body, (self.width, 0.0), (self.width, self.width), 0
            ),
            pymunk.Segment(
                self.space.static_body, (0.0, self.width), (self.width, self.width), 0
            ),
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

    def generate_data(
        self, length: int = 10000, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        states = np.zeros((length, self.num_obj, 4))
        rewards = np.zeros((length, 1))
        self.reset()

        for t in range(length):
            states[t] = self.get_state()
            if t > 0:
                states[t, :, 2:] = (states[t, :, :2] - states[t - 1, :, :2]) / dt
            self.step(dt=dt)

        return states, rewards

    def visualize(self, state: np.ndarray, save_path: Optional[str] = None):
        """Visualize a single state.

        Args:
            state: The full
            save_path: Path to save the image to.
        """
        colors = ["b", "g", "r", "c", "m", "y", "k"]
        pos = state[:, :2]
        momenta = state[:, 2:]
        fig, ax = plt.subplots(figsize=(6, 6))
        box = plt.Rectangle(
            (0, 0), self.width, self.width, linewidth=5, edgecolor="k", facecolor="none"
        )
        ax.add_patch(box)
        for i in range(self.num_obj // 2):
            circle = plt.Circle(
                (pos[i, 0], pos[i, 1]), radius=self.radii[i], edgecolor="b"
            )
            label = ax.annotate(
                "{}".format(i), xy=(pos[i, 0], pos[i, 1]), fontsize=8, ha="center"
            )
            # Plot the momentum
            plt.arrow(pos[i, 0], pos[i, 1], momenta[i, 0], momenta[i, 1])
            ax.add_patch(circle)

        for i in range(self.num_obj // 2, self.num_obj):
            circle = plt.Rectangle(
                (pos[i, 0], pos[i, 1]), self.radii[i], self.radii[i], edgecolor="b"
            )
            label = ax.annotate(
                "{}".format(i), xy=(pos[i, 0], pos[i, 1]), fontsize=8, ha="center"
            )
            # Plot the momentum
            plt.arrow(pos[i, 0], pos[i, 1], momenta[i, 0], momenta[i, 1])
            ax.add_patch(circle)
        plt.axis([0, self.width, 0, self.width])
        plt.axis("off")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def detect_collisions(self, trajectories: np.ndarray) -> np.ndarray:
        n = trajectories.shape[0]
        k = self.num_obj
        radii = np.copy(self.radii)
        radii[self.num_obj // 2 :] = radii[self.num_obj // 2 :] * np.sqrt(2)
        min_dist = radii.reshape(k, 1) + radii.reshape(1, k)
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


class WithTypes(MultiTyped):
    """Include the type of the object in the state."""

    cls_name = "with_types"

    def __init__(
        self, num_obj=5, mass: float = 10, radii: float = 10, width: float = 400
    ):
        super().__init__(num_obj, mass, radii, width)
        self.obj_dim = 5
        self.location_indices = (0, 1)

    def generate_data(
        self, length: int = 10000, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        states, rewards = super().generate_data(length, dt)

        with_types = np.zeros((length, self.num_obj, self.obj_dim))
        with_types[:, :, :-1] = states
        with_types[:, self.num_obj // 2 :, -1] = 1

        return with_types, rewards

    def detect_collisions(self, trajectories: np.ndarray) -> np.ndarray:
        return super().detect_collisions(trajectories[:, :, :-1])
