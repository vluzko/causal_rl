"""The causal environment interface."""
import numpy as np
import torch
import multiprocessing


from contextlib import contextmanager
from einops import rearrange
from functools import partial
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any

from causal_rl.config import DEVICE


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


class CausalEnv:
    """An RL environment with causal graph annotations.
    The key distinction is that the environment keeps track of causal relationships between components of states, and these can be accessed.
    """

    num_obj: int
    obj_dim: int
    use_cache: bool

    def __init__(self, num_obj, obj_dim, use_cache: bool = False) -> None:
        super().__init__()
        self.num_obj = num_obj
        self.obj_dim = obj_dim
        self.use_cache = use_cache

    def step(self) -> np.ndarray:
        """Step a causal environment forward."""
        raise NotImplementedError

    def generate_data(
        self,
        length: int,
        batch_size: int,
        num_sims: int = 1,
        num_process: int = 1,
        config_dict={},
    ) -> DataLoader:
        raise NotImplementedError

    def graph_predictor_dataloader(
        self,
        length: int,
        batch_size: int,
        *args,
        num_process: int = 1,
        num_sims: int = 1,
        **kwargs,
    ) -> DataLoader:
        raise NotImplementedError

    def next_state_dataloader(
        self,
        length: int,
        batch_size: int,
        *args,
        num_process: int = 1,
        num_sims: int = 1,
        **kwargs,
    ) -> DataLoader:
        raise NotImplementedError


class TrivialEnv(CausalEnv):
    """An environment that simply swaps the position of the objects deterministically"""

    num_dim = 2
    obj_dim = 2

    def __init__(self):
        super().__init__()
        self.num_obj = 2

    @property
    def data_folder_name(self) -> str:
        return f"trivialenv_{self.num_obj}"

    def arrange_for_attn(self, feature: np.ndarray) -> np.ndarray:
        return feature

    def multi_proc_sample_trajectory(
        self,
        length: int,
        num_sims: int,
        num_process: int,
        config_dict: Dict[str, Any] = {},
    ):
        if num_sims == 1:
            feature, edges = self.sample_trajectory(length, config_dict)
            stacked_feature = feature.reshape(1, *feature.shape)
            stacked_edge = edges.reshape(1, *edges.shape)
        else:
            with poolcontext(processes=num_process) as pool:
                bound_method = partial(
                    self.sample_trajectory, length=length, config_dict=config_dict
                )
                results = pool.map(bound_method, range(num_sims))
            feature_all = [result[0] for result in results]
            edges_all = [result[1] for result in results]

            stacked_feature = np.stack(feature_all)
            stacked_edge = np.stack(edges_all)

        return stacked_feature, stacked_edge

    def sample_trajectory(self, length: int, config_dict: Dict[str, Any]):
        """Sample a single trajectory"""
        states = np.empty((length, self.num_obj, self.obj_dim))
        states[0] = np.random.normal(0, 10, (self.num_obj, self.obj_dim))
        for i in range(1, length):
            states[i, 0] = states[i - 1, 1]
            states[i, 1] = states[i - 1, 0]
        edges = np.zeros((length, self.num_obj, self.num_obj))
        edges[:, 1, 0] = 1
        edges[:, 0, 1] = 1
        return states, edges

    def generate_data(
        self,
        length: int,
        num_sims: int = 1,
        num_process: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        features, edges = self.multi_proc_sample_trajectory(
            length, num_sims, num_process, {"sample_freq": 1}
        )

        features = rearrange(features, "s t d n -> (t s) d n")
        edges = rearrange(edges, "s t d n -> (t s) d n")

        return features, edges

    def graph_predictor_dataloader(self, length: int, batch_size: int, num_process: int = 1, num_sims: int = 1) -> DataLoader:  # type: ignore

        features, edges = self.generate_data(length, num_sims, num_process)

        feat_train = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        edges_train = torch.tensor(edges, dtype=torch.float32).to(DEVICE)

        train_data = TensorDataset(feat_train, edges_train)
        train_data_loader = DataLoader(train_data, batch_size=batch_size)
        return train_data_loader

    def next_state_dataloader(self, length: int, batch_size: int, num_process: int = 1, num_sims: int = 1) -> DataLoader:  # type: ignore
        features, edges = self.generate_data(length, num_sims, num_process)

        input_features = torch.tensor(features[:-1, :, :], dtype=torch.float32).to(
            DEVICE
        )
        target_features = torch.tensor(features[1:, :, :], dtype=torch.float32).to(
            DEVICE
        )
        edges_train = torch.tensor(edges[:-1, :, :], dtype=torch.float32).to(DEVICE)

        train_data = TensorDataset(input_features, target_features, edges_train)

        train_data_loader = DataLoader(train_data, batch_size=batch_size)

        return train_data_loader


class ConstantEnv(TrivialEnv):
    """An environment that simply swaps the position of the objects deterministically"""

    num_dim = 2
    obj_dim = 2

    def __init__(self):
        super().__init__()
        self.num_obj = 2

    @property
    def data_folder_name(self) -> str:
        return f"constantenv_{self.num_obj}"

    def sample_trajectory(self, length: int, config_dict: Dict[str, Any]):
        """Sample a single trajectory"""
        states = np.ones((length, self.num_obj, self.obj_dim))

        edges = np.zeros((length, self.num_obj, self.num_obj))
        return states, edges
