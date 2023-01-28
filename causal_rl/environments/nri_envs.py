from genericpath import exists
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import torch

from scipy.spatial import distance
from einops import rearrange
from functools import partial
from contextlib import contextmanager
from pathlib import Path
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any, Union

from causal_rl.environments import CausalEnv
from causal_rl.config import DATA, DEVICE


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


class NRISimPhysics(CausalEnv):
    @property
    def data_folder_name(self) -> str:
        raise NotImplementedError

    def run_folder(self, length: int) -> Path:
        path = DATA / f"{self.data_folder_name}_{length}"
        path.mkdir(exist_ok=True, parents=True)
        return path

    def sample_traj_wrapper(self, _iter, T, config_dict):
        return self.sample_trajectory(T, config_dict)

    def multi_proc_sample_trajectory(
        self,
        length: int,
        num_sims: int,
        num_process: int,
        config_dict: Dict[str, Any] = {},
    ):
        if num_sims == 1:
            loc, vel, edges = self.sample_trajectory(length, config_dict)
            stacked_loc = loc.reshape(1, *loc.shape)
            stacked_vel = vel.reshape(1, *vel.shape)
            stacked_edge = edges.reshape(1, *edges.shape)
        else:
            with poolcontext(processes=num_process) as pool:
                bound_method = partial(
                    self.sample_traj_wrapper, T=length, config_dict=config_dict
                )
                results = pool.map(bound_method, range(num_sims))
            loc_all = [result[0] for result in results]
            vel_all = [result[1] for result in results]
            edges_all = [result[2] for result in results]

            stacked_loc = np.stack(loc_all)
            stacked_vel = np.stack(vel_all)
            stacked_edge = np.stack(edges_all)

        if config_dict["set_max_min"]:
            self.loc_max = stacked_loc.max()
            self.loc_min = stacked_loc.min()
            self.vel_max = stacked_vel.max()
            self.vel_min = stacked_vel.min()

        # Normalize to [-1, 1]
        stacked_loc = (stacked_loc - self.loc_min) * 2 / (
            self.loc_max - self.loc_min
        ) - 1
        stacked_vel = (stacked_vel - self.vel_min) * 2 / (
            self.vel_max - self.vel_min
        ) - 1

        return stacked_loc, stacked_vel, stacked_edge

    def sample_trajectory(
        self, length: int, config_dict: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a single trajectory"""
        raise NotImplementedError

    def arrange_for_attn(
        self, feature: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return rearrange(feature, "ts d n -> ts n d")

    def graph_predictor_dataloader(
        self,
        length: int,
        batch_size: int,
        num_process: int = 1,
        num_sims: int = 1,
        set_max_min: bool = True,
        attention: bool = False,
        upsample: bool = False,
    ) -> DataLoader:
        assert length % 2 == 0

        loc, vel, edge = self.multi_proc_sample_trajectory(
            length,
            num_sims,
            num_process,
            {"sample_freq": 1, "set_max_min": set_max_min},
        )

        # Concat position and velocity together into DOF
        feature = np.concatenate([loc, vel], axis=2)

        # Make time the first axis, group batch and time together
        feature = rearrange(feature, "s t d n -> (t s) d n")
        edge = rearrange(edge, "s t d n -> (t s) d n")
        if attention:
            feature = self.arrange_for_attn(feature)
        else:
            feature = rearrange(feature, "ts d n -> ts (d n)")

        if upsample:
            # The indices of the states that have a non empty graph
            indices = np.unique(np.nonzero(edge)[0])
            feature = feature[indices]
            edge = edge[indices]

        feature = torch.tensor(feature, dtype=torch.float32).to(DEVICE)
        target = torch.tensor(edge, dtype=torch.float32).to(DEVICE)

        train_data = TensorDataset(feature, target)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_data_loader

    def next_state_dataloader(
        self,
        length: int,
        num_sims: int,
        batch_size: int,
        num_process: int,
        set_max_min: bool,
        attention: bool = False,
        upsample: bool = False,
    ) -> DataLoader:
        assert length % 2 == 0
        loc, vel, edges_np = self.multi_proc_sample_trajectory(
            length,
            num_sims,
            num_process,
            {"sample_freq": 1, "set_max_min": set_max_min},
        )
        features_np = np.concatenate([loc, vel], axis=3)

        if attention:
            features_np = self.arrange_for_attn(features_np)
        else:
            features_np = rearrange(features_np, "s t d n -> (s t) (d n)")

        edges_np = rearrange(edges_np, "s t n k -> (s t) n k")

        if upsample:
            # The indices of the states that have a non empty graph
            indices = np.unique(np.nonzero(edges_np)[0])
            features_np = features_np[indices]
            edges_np = edges_np[indices]

        # TODO: Only move once
        features = torch.tensor(features_np[:-1], dtype=torch.float32).to(DEVICE)
        targets = torch.tensor(features_np[1:], dtype=torch.float32).to(DEVICE)
        edges = torch.tensor(edges_np[:-1], dtype=torch.float32).to(DEVICE)

        if attention:
            features = self.arrange_for_attn(features)  # type: ignore
            targets = self.arrange_for_attn(targets)  # type: ignore

        train_data = TensorDataset(features, targets, edges)  # type: ignore
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_data_loader

    def generate_data(
        self,
        length: int,
        batch_size: int = 64,
        num_sims: int = 1,
        num_process: int = 1,
        config_dict: Dict[str, Any] = {"set_max_min": True},
    ):  # type: ignore
        states_path = self.run_folder(length) / "states.tch"
        edges_path = self.run_folder(length) / "edges.tch"
        if self.use_cache and states_path.is_file() and edges_path.is_file():
            states = torch.load(str(states_path)).to(DEVICE)
            edges = torch.load(str(edges_path)).to(DEVICE)
        else:
            loc, vel, edges = self.multi_proc_sample_trajectory(
                length, num_sims, num_process, config_dict
            )
            num_atoms = loc.shape[3]

            # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
            states_np = np.concatenate([loc, vel], axis=2)
            states_np = rearrange(states_np, "s t d a -> (s t) a d")
            edges_np = np.reshape(edges, [-1, num_atoms**2])
            edges_np = np.array((edges_np + 1) / 2, dtype=np.int64)

            states = torch.from_numpy(states_np).float().to(DEVICE)
            edges = torch.from_numpy(edges_np).float().to(DEVICE)

            # Exclude self edges
            off_diag_idx = np.ravel_multi_index(
                np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
                [num_atoms, num_atoms],
            )
            edges = edges[:, off_diag_idx]
            torch.save(states, states_path)
            torch.save(edges, edges_path)

        train_data = TensorDataset(states[:-1], edges[:-1], states[1:])

        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        return train_data_loader


class HardSpheres(NRISimPhysics):
    num_dim = 4
    obj_dim = 4

    def __init__(
        self,
        num_obj: int = 5,
        mass: float = 10.0,
        radius: float = 10.0,
        width: float = 150.0,
        dt: float = 0.01,
        use_cache: bool = False,
    ):
        super().__init__(num_obj, 4, use_cache=use_cache)
        self.num_obj = num_obj
        self.mass = mass
        self.radius = radius
        self.masses = mass * np.ones(num_obj)
        self.radii = radius * np.ones(num_obj)
        self.width = width
        self.location_indices = (0, 1)
        self.dt = dt

    @property
    def data_folder_name(self) -> str:
        return f"hardspheres_{self.num_obj}"

    def reset(self):
        """Creates Pymunk environment for simulation of spheres"""
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
        return self.get_state()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the location and positions of the spheres
        :param loc: Nx2 location at one time stamp
        :param vel: Nx2 velocity at one time stamp
        """
        loc = np.zeros((self.num_obj, 2))
        vel = np.zeros((self.num_obj, 2))
        for i in range(self.num_obj):
            loc[i] = np.array(
                [self.objects[i].position[0], self.objects[i].position[1]]
            )
            vel[i] = np.array(
                [self.objects[i].velocity[0], self.objects[i].velocity[1]]
            )
        return loc, vel

    def step(self) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        self.space.step(self.dt)
        return self.get_state()

    def generate_positions(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates locs, vels for the full trajectory but not yet edges

        Args:
            T:
            sample_freq:
        """
        locs = np.zeros((T, self.num_obj, 2))
        vels = np.zeros((T, self.num_obj, 2))
        # assert T % sample_freq == 0
        # T_save = int(T / sample_freq - 1)
        counter = 0
        locs_red = np.zeros((T, self.num_obj, 2))
        vels_red = np.zeros((T, self.num_obj, 2))
        self.reset()

        for t in range(T):
            locs[t], vels[t] = self.get_state()
            if t > 0:
                vels[t, :, :2] = (locs[t, :, :2] - locs[t - 1, :, :2]) / self.dt
            if t > 0:
                locs_red[counter] = locs[t]
                vels_red[counter] = vels[t]
                counter += 1
            self.step()
        return locs_red, vels_red

    def visualize(self, state: np.ndarray, save_path: Optional[str] = None):
        """Visualize a single state.

        Args:
            state: State at a single time step.
            save_path: Where to save the snapshot
        """
        pos = state[:, :2]
        momenta = state[:, 2:]
        _, ax = plt.subplots(figsize=(6, 6))
        box = plt.Rectangle((0, 0), self.width, self.width, linewidth=5, edgecolor="k", facecolor="none")  # type: ignore
        ax.add_patch(box)
        for i in range(len(pos)):
            circle = plt.Circle((pos[i, 0], pos[i, 1]), radius=self.radii[i], edgecolor="b")  # type: ignore
            _label = ax.annotate(
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

    def detect_collisions(self, locs: np.ndarray) -> np.ndarray:
        """given the full trajectory of locations, determines the collision matrix, analog of edges"""
        T = locs.shape[0]
        k = self.num_obj
        min_dist = self.radii.reshape(k, 1) + self.radii.reshape(1, k)
        np.fill_diagonal(min_dist, 0)

        collisions = np.zeros((T, k, k))

        for t in range(1, T):
            # The (x, y) coordinates of all balls at t=i
            loc = locs[t, :, :2]

            distances = distance.squareform(distance.pdist(loc))
            collided = np.nonzero(distances < min_dist)

            collisions[t - 1][collided] = 1

            collisions[t][collided] = 1

        return collisions

    def sample_trajectory(
        self, T: int, config_dict: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # sample_freq = config_dict["sample_freq"]
        locs, vels = self.generate_positions(T)
        edges = self.detect_collisions(locs)
        # NRI format wants (Timesteps/sample_freq -1, Degrees of Freedom, Number of Balls)
        locs = np.einsum("ijk->ikj", locs)
        vels = np.einsum("ijk->ikj", vels)
        # edges im averaging because it predicts static graphs
        # edges = np.einsum('ijk->jk',edges)/T
        return locs, vels, edges


class DenseEnv(NRISimPhysics):
    """An environment with fairly dense causal graphs"""

    num_dim = 4
    obj_dim = 4

    def __init__(
        self,
        num_obj: int,
        min_radius: float = 0.05,
        max_radius: float = 0.35,
        dt: float = 0.01,
        use_cache: bool = False,
    ):
        super().__init__(num_obj, 4, use_cache=use_cache)
        self.num_obj = num_obj
        self.dt = dt
        self.min_radius = min_radius
        self.max_radius = max_radius

    @property
    def data_folder_name(self) -> str:
        return f"denseenv_{self.num_obj}"

    def arrange_for_attn(self, feature: np.ndarray) -> np.ndarray:  # type: ignore
        return feature

    def sample_trajectory(self, length: int, config_dict: Dict[str, Any]):
        """Sample a single trajectory"""
        state = np.zeros((self.num_obj, self.obj_dim))
        # Toroidal space
        state[:, :2] = np.random.uniform(0, 1, (self.num_obj, 2))
        state[:, 2:] = np.random.uniform(-1, 1, (self.num_obj, 2))
        states = np.zeros((length, self.num_obj, self.obj_dim))
        graphs = []
        for i in range(length):
            state[:, :2] += state[:, 2:] * self.dt
            state[:, :2] %= 1

            # apply forces
            distances = distance.squareform(distance.pdist(state[:, :2]))
            in_range = (self.min_radius < distances) & (distances < self.max_radius)
            contributions = in_range @ state[:, 2:] * 0.1

            state[:, 2:] += contributions

            states[i] = state
            state[:, 2:] %= 10
            graphs.append(in_range)
        locs = states[:, :, :2]
        vels = states[:, :, 2:]
        edges = np.array(graphs)
        return locs, vels, edges
