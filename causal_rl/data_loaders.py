import numpy as np
import torch

from pathlib import Path
from random import sample
from torch.utils.data import Dataset
from typing import Tuple, List, Union, Set

from causal_rl.environments import CausalEnv
from causal_rl.utils import load_sim


class StateAndGraph(Dataset):
    """Abstract base class.
    Does absolutely nothing, just here for type checking.
    """

    def __init__(self, env: CausalEnv):
        super().__init__()
        self.env = env

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        raise NotImplementedError

    def update(self, index: Union[int, torch.Tensor]):
        raise NotImplementedError


class SimpleDataset(StateAndGraph):
    def __init__(self, env: CausalEnv, steps: int):
        self.env = env
        self.steps = steps
        self.states, self.collisions, self.rewards = load_sim(self.env, self.steps + 1)
        self.indices: List[int] = []
        self.bad_state_indices: Set[int] = set()

    def update(self, index: Union[int, torch.Tensor]):
        if isinstance(index, int):
            self.bad_state_indices.add(index)
        else:
            for i in index:
                self.bad_state_indices.add(i.item())

    def __len__(self):
        return self.steps

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        self.indices.append(index)
        return self.states[index], self.states[index + 1], self.collisions[index], index


class BufferedState(SimpleDataset):
    def __init__(self, env, steps: int, prob: float, buffer_size: int = 200):
        super().__init__(env, steps)
        self.prob = prob
        self.buffer_size = buffer_size

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if (
            len(self.bad_state_indices) > self.buffer_size
        ) and np.random.random() < self.prob:
            new_index = sample(self.bad_state_indices, 1)[0]
        else:
            new_index = index
        self.indices.append(new_index)
        return (
            self.states[new_index],
            self.states[new_index + 1],
            self.collisions[new_index],
            new_index,
        )


class PrebufferedState(BufferedState):
    def __init__(
        self,
        env: CausalEnv,
        steps: int,
        prob: float,
        bad_states: torch.Tensor,
        bad_collisions: torch.Tensor,
    ):
        super().__init__(env, steps, prob)
        self.base_states = self.states
        self.base_collisions = self.collisions

        self.bad_states = bad_states
        self.bad_collisions = bad_collisions

        self.states = torch.cat((self.base_states, self.bad_states))
        self.collisions = torch.cat((self.base_collisions, self.bad_collisions))
        self.offset = len(self.base_states)
        self.bad_count = 0

    def update(self, index: Union[int, torch.Tensor]):
        if isinstance(index, int):
            self.bad_state_indices.add(index)
        else:
            for i in index:
                self.bad_state_indices.add(i.item())

    def __len__(self):
        return self.steps

    def __getitem__(self, index: int):
        if np.random.random() < self.prob:
            temp = np.random.choice(len(self.bad_states) - 1)
            real_index = self.offset + temp
            self.bad_count += 1
        else:
            real_index = index % (self.offset - 1)
        self.indices.append(real_index)
        return (
            self.states[real_index],
            self.states[real_index + 1],
            self.collisions[real_index],
            real_index,
        )


class StoredBuffer(StateAndGraph):
    """An alternative version of PrebufferedState that can be stored.
    The implementation isn't clean but it should be easy to understand and modify.
    Eventually this will replace PrebufferedState entirely.
    """

    def __init__(self, env: CausalEnv, prob: float, generator: "Experiment"):
        super().__init__(env)
        self.bad_state_indices: Set[int] = set()
        try:
            self.states, self.collisions = self.load()
            self.steps = len(self.states) - 1
        except FileNotFoundError:
            bad_states, bad_collisions = self.generate_bad_states(generator)
            extra_steps = int(len(bad_states) / prob)
            good_states, good_collisions, _ = load_sim(self.env, extra_steps + 1)
            self.steps = len(bad_states) + extra_steps

            # Plausibly these should be shuffled so it still works when you don't pass shuffle=True.
            self.states = torch.cat((good_states, bad_states))
            self.collisions = torch.cat((good_collisions, bad_collisions))
            self.store()

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int):
        return self.states[index], self.states[index + 1], self.collisions[index], index

    def update(self, index: Union[int, torch.Tensor]):
        if isinstance(index, int):
            self.bad_state_indices.add(index)
        else:
            for i in index:
                self.bad_state_indices.add(i.item())

    def generate_bad_states(self, generator) -> Tuple[torch.Tensor, torch.Tensor]:
        generator.run()
        bad_states = generator.dataset.states[list(generator.dataset.bad_state_indices)]
        bad_collisions = generator.dataset.collisions[
            list(generator.dataset.bad_state_indices)
        ]

        return bad_states, bad_collisions

    def paths(self) -> Tuple[Path, Path]:
        base_name = "stored_buffer_{}_".format(self.env.name)
        states_path = DATA / (base_name + "states.tch")
        collisions_path = DATA / (base_name + "collisions.tch")
        return states_path, collisions_path

    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        states_path, collisions_path = self.paths()

        states = torch.load(str(states_path))
        collisions = torch.load(str(collisions_path))
        return states, collisions

    def store(self):
        states_path, collisions_path = self.paths()

        torch.save(self.states, str(states_path))
        torch.save(self.collisions, str(collisions_path))


class Trajectories(Dataset):
    """For use with any architectures expecting trajectories instead of individual samples."""

    def __init__(self, env: CausalEnv, steps: int, trajectory_size: int = 10):
        self.env = env
        self.steps = steps
        self.raw_states, self.raw_collisions, _ = load_sim(
            self.env, (self.steps + 1) * trajectory_size
        )
        self.states = self.raw_states.view(
            -1, trajectory_size, self.raw_states.shape[1], self.raw_states.shape[2]
        )
        # NRI expects objects x steps x dimensions
        self.states = self.states.permute(0, 2, 1, 3)
        self.collisions = self.raw_collisions.view(
            -1,
            trajectory_size,
            self.raw_collisions.shape[1],
            self.raw_collisions.shape[2],
        )
        self.bad_state_indices: Set[int] = set()

    def __len__(self):
        return self.steps

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.states[index], self.states[index + 1], self.collisions[index], index
