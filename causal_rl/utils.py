import numpy as np
import torch

from matplotlib import pyplot as plt
from pathlib import Path
from torch.nn import functional as F
from typing import Tuple, Union, Any

from causal_rl import tracker, config


def get_tracker(keyword: config.TrackerKey, project_name: str, config: Any):
    """Returns a tracker class for logging, see tracker.py"""
    if keyword == "wandb":
        return tracker.WandBTracker(project_name, config)
    elif keyword == "console":
        return tracker.ConsoleTracker(project_name, config)
    else:
        raise ValueError("Not an accepted tracker")


def sample_gumbel(shape: torch.Tensor, eps=1e-20) -> torch.Tensor:
    """Sample from a gumbel distribution"""
    unif = torch.rand(shape).to(config.DEVICE)  # type: ignore
    return -torch.log(-torch.log(unif + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float,
    latent_dim: int,
    categorical_dim: int,
    hard: bool = False,
) -> torch.Tensor:
    """ST-gumple-softmax
    Args: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


def pairs_to_linear_index(n: int) -> torch.Tensor:
    index = 0
    pairs = torch.zeros((n, n), dtype=torch.long, device=config.DEVICE)
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs[i, j] = index
                index += 1
    return pairs


def symmetric_to_asymmetric_indices(n: int) -> torch.Tensor:
    """Map a flattened upper triangular adjacency matrix to a flattened normal matrix.
    In other words, it takes the upper triangular matrix [[*, 0, 1], [* , 2], [*]]
    and returns [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]] (skipping the diagonal).

    Args:
        n: The width of the upper triangular matrix

    Returns:
        A set of indices that will map an upper triangular matrix of the given size to
        a full matrix of the given size, minus the diagonal.

    Example 1:
        >>> symmetric_to_asymmetric_indices(3)
        tensor([0, 1, 0, 2, 1, 2])

    Example 2:
        >>> x = torch.arange(1, 4)
        >>> y = symmetric_to_asymmetric_indices(3)
        >>> x[y]
        tensor([1, 2, 1, 3, 2, 3])

    """
    index = 0
    flat = torch.zeros(n * (n - 1), dtype=torch.long, device=config.DEVICE)

    for i in range(n):
        for j in range(i + 1, n):
            if i != j:
                flat[i * (n - 1) + j - 1] = index
                flat[j * (n - 1) + i] = index
                index += 1
    return flat


def upper_triangular_indices(n: int) -> list:
    indices = []

    for i in range(n):
        new_indices = [[i, x] for x in range(i + 1, n)]
        indices.extend(new_indices)

    return indices


def state_indices(n: int) -> torch.Tensor:
    """Create an index map to expand a state tensor into half of a (state, state) pairs tensor
    Useful when you need to do computations on all pairs of states instead of all states.
    """
    indices = torch.zeros(n * (n - 1), dtype=torch.long, device=config.DEVICE)
    for i in range(n):
        place = 0
        for j in range(n):
            if i != j:
                indices[(i * (n - 1)) + place] = j
                place += 1

    return indices


def state_to_source_sink_indices(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create an index map to expand a state tensor into all (non-identical) pairs of states.

    Args:
        n: The number of nodes.

    Example:
        If we have 3 nodes, output will be:
            [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]
    """
    right_indices = torch.zeros(n * (n - 1), dtype=torch.long, device=config.DEVICE)
    for i in range(n):
        place = 0
        for j in range(n):
            if i != j:
                right_indices[(i * (n - 1)) + place] = j
                place += 1
    left_indices = torch.repeat_interleave(torch.arange(n), n - 1, dim=0).to(
        config.DEVICE
    )
    return left_indices, right_indices


def flattened_upper_adj_to_adj_indices(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create an index map to expand a state tensor into all unordered pairs of states

    Returns:
        Two index arrays. The first extracts the sources from the flattened matrix
            the second extracts the sinks.

    Example:
        If we have 3 vertices, our flattened upper triangular matrix is of size 3,
        and the entries correspond to [0, 1], [0, 2], [1, 2]

        This would produce:
        [0, 0, 1], [1, 2, 2]

    """
    sources = torch.zeros(int(n * (n - 1) / 2), dtype=torch.long)
    sinks = torch.zeros(int(n * (n - 1) / 2), dtype=torch.long)
    r = torch.arange(1, n)
    starts = torch.zeros(n - 1, dtype=torch.int)
    start = 0
    for i in range(1, n):
        starts[i - 1] = start
        count = n - i
        sources[start : start + count] = i - 1
        sinks[start : start + count] = r[i - 1 :]
        start += count

    return sources.to(config.DEVICE), sinks.to(config.DEVICE)


def store_sim(env, steps) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run and store a simulation of the given environment."""
    env_name = env.name
    env_size = env.num_obj
    states, rewards = env.generate_data(length=steps)
    collisions = env.detect_collisions(states)

    s_path = config.DATA / "{}_{}_{}_states.npy".format(env_name, env_size, steps)
    c_path = config.DATA / "{}_{}_{}_graphs.npy".format(env_name, env_size, steps)
    r_path = config.DATA / "{}_{}_{}_rewards.npy".format(env_name, env_size, steps)

    config.DATA.mkdir(parents=True, exist_ok=True)

    np.save(s_path.open("wb"), states)
    np.save(c_path.open("wb"), collisions)
    np.save(r_path.open("wb"), rewards)
    return states, collisions, rewards


def load_sim(
    env, steps: int, gen_new: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a simulation of the given environment with the given number of steps.
    If no such simulation exists or gen_new is True, then a new simulation is run.
    """
    if gen_new:
        store_sim(env, steps)

    env_name = env.name
    env_size = env.num_obj
    s_path = config.DATA / "{}_{}_{}_states.npy".format(env_name, env_size, steps)
    c_path = config.DATA / "{}_{}_{}_graphs.npy".format(env_name, env_size, steps)
    r_path = config.DATA / "{}_{}_{}_rewards.npy".format(env_name, env_size, steps)

    if not (s_path.exists() and c_path.exists()):
        states, graphs, rewards = store_sim(env, steps)
    else:
        states = np.load(s_path.open("rb"))
        graphs = np.load(c_path.open("rb"))
        rewards = np.load(r_path.open("rb"))

    t_states = torch.from_numpy(states).float().to(config.DEVICE)
    t_graphs = torch.from_numpy(graphs).float().to(config.DEVICE)
    t_rewards = torch.from_numpy(rewards).float().to(config.DEVICE)

    return t_states, t_graphs, t_rewards


def plot_exploration(
    losses: np.ndarray,
    exp_name: str,
    env_name: str,
    save: bool = True,
    add_title: bool = True,
):
    """Create a plot for an exploratory experiment."""
    fig, ax = plt.subplots()

    ax.plot(losses, linestyle="None", marker="o", markersize=1)

    upper = np.ceil(np.log10(losses.max()))
    ax.set_ylim([0, 10**upper])  # type: ignore
    ax.set_xlabel("Step")
    ax.set_ylabel("L2 Loss")
    if add_title:
        ax.set_title("{} on {}".format(exp_name, env_name))

    if save:
        folder = config.PLOTS / "tmp"
        folder.mkdir(parents=True, exist_ok=True)
        save_path = folder / "{}_{}.png".format(exp_name, env_name)
        plt.savefig(save_path)
    else:
        plt.show()


def plot_display(losses: np.ndarray, folder: Path, title: str, upper: int):
    """Create a plot for display."""
    fig, ax = plt.subplots()

    ax.plot(losses, linestyle="None", marker="o", markersize=1)

    ax.set_ylim([0, upper])  # type: ignore
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    save_folder = config.PLOTS / folder
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = config.PLOTS / folder / "{}.png".format(title)
    plt.savefig(save_path)


def plot_state_loss(
    losses: Union[np.ndarray, torch.Tensor],
    folder: Union[str, Path],
    title: str,
    upper: int = 10000,
):
    fig, ax = plt.subplots()

    ax.plot(losses, linestyle="None", marker="o", markersize=1)

    ax.set_ylim([0, upper])  # type: ignore
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Overall loss")

    save_folder = config.PLOTS / folder
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = config.PLOTS / folder / "{}.png".format(title)
    plt.savefig(save_path)


def plot_coll_loss(
    losses: Union[np.ndarray, torch.Tensor],
    folder: Union[str, Path],
    title: str,
    upper: int = 20,
):
    fig, ax = plt.subplots()

    ax.plot(losses, linestyle="None", marker="o", markersize=1)

    ax.set_ylim([0, upper])  # type: ignore
    ax.set_xlabel("Step")
    ax.set_ylabel("L2 Loss")
    ax.set_title("Collision loss")

    save_folder = config.PLOTS / folder
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = config.PLOTS / folder / "coll_{}.png".format(title)
    plt.savefig(save_path)


def plot_grad(grads: torch.Tensor, folder: Union[Path, str], title: str):
    """Plot gradients."""
    fig, ax = plt.subplots()
    ax.plot(grads, linestyle="None", marker="o", markersize=3)

    ax.set_ylim([0, 10000])  # type: ignore
    ax.set_xlabel("Batch")
    ax.set_ylabel("Gradient Norm")

    save_folder = config.PLOTS / folder
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / "{}.png".format(title)
    plt.savefig(save_path)
    plt.close()
