import numpy as np
import torch

from fire import Fire
from matplotlib import pyplot as plt
from pathlib import Path
from random import sample
from torch import optim, distributions
from torch.optim.optimizer import Optimizer
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Union, Set

from causal_rl.environments import CausalEnv, HardSpheres, MultiTyped, WithTypes, Mujoco
from causal_rl.graph_predictor import FullyConnectedPredictor, ConvLinkPredictor
from causal_rl.state_predictor import DiscretePredictor, WeightedPredictor
from causal_rl.model import NaivePredictor, GraphAndMessages
from causal_rl.utils import plot_exploration, plot_coll_loss, plot_state_loss, load_sim

ENVIRONMENT_MAP = {
    "bouncing_balls": HardSpheres,
    "multi_typed": MultiTyped,
    "with_types": WithTypes,
    "mujoco": Mujoco,
}
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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

    def __getitem__(self, index: int):
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


# Keys are (convolutional, variational, local)
graph_predictor_map = {
    False: FullyConnectedPredictor,
    True: ConvLinkPredictor,
}


class Experiment:

    name = ""

    def __init__(self, env, steps: int = 100, lr: float = 0.001, gen_new: bool = False):
        self.env = env
        self.steps = steps
        self.lr = lr
        self.gen_new = gen_new
        self.writer = SummaryWriter()

    @property
    def display_name(self):
        return "{}_{}".format(self.name, self.steps)

    @property
    def hyperparameter_dict(self):
        return {"env": self.env.name, "steps": self.steps, "lr": self.lr}

    @property
    def hyperparameter_str(self):
        return "env_{}_steps_{}_lr_{}".format(self.env.name, self.steps, self.lr)

    @classmethod
    def hyperparameter_space(cls):
        """Return the space of all hyperparameters"""
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class NaiveModel(Experiment):
    """Predict the next state with a naive model that depends only on the current state."""

    name = "naive_model"

    def __init__(
        self,
        env,
        steps: int,
        lr: float,
        epochs: int = 1,
        gen_new: bool = True,
        reg_param: float = 0.5,
        batch_size: int = 32,
        save_model: bool = False,
    ):
        super().__init__(env, steps, lr, gen_new=gen_new)

        self.save_model = save_model

        # Training parameters
        self.epochs = epochs
        self.reg_param = reg_param
        self.batch_size = batch_size

        self.state_predictor = NaivePredictor(self.env.num_obj, self.env.obj_dim).to(
            DEVICE
        )

    @property
    def graph_size(self):
        return self.env.num_obj * (self.env.num_obj - 1)

    @property
    def display_name(self):
        return "{}_{}".format(self.name, self.steps)

    def get_data(self) -> Tuple[StateAndGraph, DataLoader]:
        "Get a dataset and dataloader for the current environment"
        self.dataset = SimpleDataset(self.env, self.steps)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        return self.dataset, self.dataloader

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the entire training loop

        Returns:
            Losses and collision losses
        """
        # Setup
        dataset, dataloader = self.get_data()
        s_opt = optim.Adam(self.state_predictor.parameters(), lr=self.lr)
        s_opt.zero_grad()

        losses = torch.zeros(self.epochs * self.steps)

        for e in range(self.epochs):
            print("Epoch: {}".format(e))
            for i, (
                node_state_batch,
                true_state_batch,
                true_graph_batch,
                true_indices,
            ) in enumerate(dataloader):
                mini_batch_size = len(node_state_batch)

                storage_index = e * self.steps + i * self.batch_size

                # The raw predictions for the upper triangle of the adjacency matrix.
                state_pred = self.state_predictor(node_state_batch)

                # Calculate losses and backprop

                # MSE Loss
                loss = functional.mse_loss(
                    state_pred, true_state_batch, reduction="none"
                ).view(mini_batch_size, -1)
                batch_loss = loss.sum(dim=1) / loss.shape[1]

                # L2 norm loss
                # batch_loss = torch.norm((true_state_batch - state_pred).view(mini_batch_size, -1), dim=1)

                mean_loss = batch_loss.mean()

                # Store bad states.
                if storage_index > self.batch_size * 5:
                    max_loss = (
                        2.0
                        * losses[storage_index - self.batch_size : storage_index].mean()
                    )
                    filt = batch_loss > max_loss
                    bad_indices = true_indices[filt.nonzero(as_tuple=False).view(-1)]
                    dataset.update(bad_indices)

                mean_loss.backward()

                s_opt.step()

                s_opt.zero_grad()

                # Store everything
                batch_norm = torch.norm(
                    (true_state_batch - state_pred).view(mini_batch_size, -1), dim=1
                )
                losses[
                    storage_index : storage_index + mini_batch_size
                ] = (
                    batch_loss.detach()
                )  # / true_state_batch.view(mini_batch_size, -1).norm(dim=1)
                # self.writer.add_scalar('State L2 Loss', mean_loss.item(), storage_index)

            print(losses[e * self.steps : (e + 1) * self.steps].mean())

        if self.save_model:
            self.save()

        return losses, torch.zeros(self.epochs * self.steps)


class FullArch(Experiment):
    """The full architecture experiment.

    Attributes:
        env (Environment):                  The environment to run on.
        graph_predictor (CDAGPredictor):    A neural network whose output is a distribution on Bipartite(k, k), where k is the number of
                                                objects in the environment.
        state_predictor (nn.Module):        A neural network from (current_state, Bipartite(k, k)) -> next_state.
        variational (bool):                 Whether or not to use the weighted or the sampling architecture.
        asymmetric (bool):                  Whether or not the model should predict asymmetric causal graphs.
        convolutional (bool):               Whether or not the graph predictor should be convolutional or fully-connected
        local (bool):                       Whether or not to force the graph predictor to only predict local causal relations.
        epochs (int):                       Number of epochs to run for.
        steps (int):                        Number of steps in an epoch.
        batch_size (int):                   Size of an individual batch.
        lr (float):                         The initial learning rate for Adam.
        reg_param (float):                  L1 regularization parameter on the causal graph prediction.
    """

    name = "full"

    def __init__(
        self,
        env,
        steps: int,
        lr: float,
        epochs: int = 1,
        gen_new: bool = True,
        reg_param: float = 0.0,
        lin_widths: Tuple[int, ...] = (256, 256, 256),
        msg_widths: Tuple[int, ...] = (256, 256),
        final_widths: Tuple[int, ...] = (256, 256),
        asymmetric: bool = False,
        convolutional: bool = False,
        local: bool = False,
        iteration_steps: int = 0,
        bad_state_weight: float = 1.0,
        variational: bool = False,
        buffer_prob: float = 0.0,
        max_distance: float = 30.0,
        batch_size: int = 128,
        save_model: bool = False,
    ):
        super().__init__(env, steps, lr, gen_new=gen_new)

        self.save_model = save_model

        # Training parameters
        self.epochs = epochs
        self.reg_param = reg_param
        self.batch_size = batch_size
        self.buffer_prob = buffer_prob
        self.iteration_steps = iteration_steps
        self.bad_state_weight = bad_state_weight

        # Model parameters
        self.asymmetric = asymmetric
        self.convolutional = convolutional
        self.variational = variational
        self.local = local
        self.max_distance = max_distance

        # Make the graph predictor
        # TODO: Cleanup. Should be one if/else. Or just pass the graph and state predictors as args.
        graph_predictor_class = graph_predictor_map[convolutional]
        if self.local:
            self.graph_predictor = graph_predictor_class(
                env.num_obj,
                env.obj_dim,
                asymmetric=self.asymmetric,
                max_distance=self.max_distance,
                location_indices=self.env.location_indices,
                layer_widths=lin_widths,
            ).to(DEVICE)
        else:
            self.graph_predictor = graph_predictor_class(
                env.num_obj,
                env.obj_dim,
                asymmetric=self.asymmetric,
                layer_widths=lin_widths,
            ).to(DEVICE)

        # Make the state predictor
        if variational:
            self.state_predictor = DiscretePredictor(
                self.env.num_obj, self.env.obj_dim
            ).to(DEVICE)
        else:
            self.state_predictor = WeightedPredictor(
                self.env.num_obj,
                self.env.obj_dim,
                msg_widths=msg_widths,
                final_widths=final_widths,
            ).to(DEVICE)

    @property
    def graph_size(self):
        return self.env.num_obj * (self.env.num_obj - 1)

    @property
    def display_name(self):
        extra = ""
        if self.asymmetric:
            extra += "_asymmetric"
        if self.variational:
            extra += "_variational"
        if self.bad_state_weight != 1.0:
            extra += "_overweight({})".format(self.bad_state_weight)
        if self.iteration_steps > 0:
            extra += "_iteration({})".format(self.iteration_steps)
        if self.local:
            extra += "_local"
        if self.convolutional:
            extra += "_wconvs"
        if self.buffer_prob > 0.0:
            extra += "_buffer({})".format(self.buffer_prob)
        return "{}_{}".format(self.name, self.steps) + extra

    def get_data(self) -> Tuple[StateAndGraph, DataLoader]:
        "Get a dataset and dataloader for the current environment"
        if self.buffer_prob > 0.0:
            dataset: StateAndGraph = BufferedState(
                self.env, self.steps, self.buffer_prob
            )
        else:
            dataset = SimpleDataset(self.env, self.steps)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return self.dataset, self.dataloader

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the entire training loop

        Returns:
            Losses and collision losses
        """
        # Setup
        dataset, dataloader = self.get_data()
        g_opt = optim.Adam(self.graph_predictor.parameters(), lr=self.lr)
        s_opt = optim.Adam(self.state_predictor.parameters(), lr=self.lr)
        g_opt.zero_grad()
        s_opt.zero_grad()

        losses = torch.zeros(self.epochs * self.steps)
        pred_collisions = torch.zeros((self.epochs * self.steps, self.graph_size)).to(
            DEVICE
        )

        # The true collisions at each step
        true_collisions: List[torch.Tensor] = []

        # The gradients of the graph predictor at each step
        graph_gradients: List[List[float]] = []

        for e in range(self.epochs):
            print("Epoch: {}".format(e))

            for i, (
                node_state_batch,
                true_state_batch,
                true_graph_batch,
                true_indices,
            ) in enumerate(dataloader):
                mini_batch_size = len(node_state_batch)
                storage_index = e * self.steps + i * self.batch_size

                # TODO: Move state and graph prediction into FullModel class.
                # The raw predictions for the upper triangle of the adjacency matrix.
                pred_adj_matrix_probabilities = self.graph_predictor(
                    node_state_batch.view(mini_batch_size, -1)
                )
                if not self.variational:
                    pred_adj_matrix = pred_adj_matrix_probabilities
                else:
                    dist = distributions.Bernoulli(probs=pred_adj_matrix_probabilities)
                    pred_adj_matrix = dist.sample()

                # Next state prediction
                state_pred = self.state_predictor(node_state_batch, pred_adj_matrix)

                ## Calculate losses and backprop

                # MSE Loss
                loss = functional.mse_loss(
                    state_pred, true_state_batch, reduction="none"
                ).view(mini_batch_size, -1)
                batch_loss = loss.sum(dim=1) / loss.shape[1]

                # Handle bad states
                if storage_index > self.batch_size * 5:
                    max_loss = (
                        2.0
                        * losses[
                            storage_index - self.batch_size // 2 : storage_index
                        ].mean()
                    )
                    filt = batch_loss > max_loss

                    # Store bad state indices
                    bad_indices = true_indices[filt.nonzero(as_tuple=False).view(-1)]
                    dataset.update(bad_indices)

                    # Overweight bad states
                    batch_loss[filt] *= self.bad_state_weight

                mean_loss = batch_loss.mean()

                # L1 regularization for the graph predictor
                if self.reg_param > 0:
                    l1_reg = self.reg_param * torch.norm(
                        pred_adj_matrix_probabilities, p=1
                    )
                    combined_loss = l1_reg + mean_loss
                else:
                    combined_loss = mean_loss

                combined_loss.backward(retain_graph=True)

                # The loss for the graph predictor
                # Only defined when we use the variational architecture.
                if self.variational:
                    graph_loss = dist.log_prob(pred_adj_matrix) * combined_loss.detach()
                    graph_loss.mean().backward()

                # TODO: Move into FullModel class.
                # Record graph predictor gradients
                # for j in (0, 3, 5):
                #     layer = self.graph_predictor.layers[j]
                #     grad_norm = layer.weight.grad.norm().item()
                #     self.writer.add_scalar('Graph Predictor Layer {} Gradient'.format(j // 2), grad_norm, i)

                self.backprop(g_opt, s_opt)

                # Store everything
                pred_collisions[
                    storage_index : storage_index + mini_batch_size
                ] = pred_adj_matrix_probabilities
                true_collisions.append(true_graph_batch)
                batch_norm = torch.norm(
                    (true_state_batch - state_pred).view(mini_batch_size, -1), dim=1
                )
                losses[
                    storage_index : storage_index + mini_batch_size
                ] = (
                    batch_loss.detach()
                )  # / true_state_batch.view(mini_batch_size, -1).norm(dim=1)
                self.writer.add_scalar(
                    "State L2 Loss", combined_loss.item(), storage_index
                )
            print(self.test())

        if self.save_model:
            self.save()

        # Compute collision losses
        collision_losses, full_coll_losses = self.collision_losses(
            true_collisions, pred_collisions
        )
        # for i, cl in enumerate(collision_losses):
        #     self.writer.add_scalar('Collision L2 Loss', cl, i)

        # Store gradients
        # grad_tens = torch.tensor(graph_gradients)

        return losses, full_coll_losses.detach()

    def backprop(self, g_opt: Optimizer, s_opt: Optimizer):
        g_opt.step()
        s_opt.step()
        g_opt.zero_grad()
        s_opt.zero_grad()

    def collision_losses(
        self, true_collisions: List[torch.Tensor], pred_collisions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the L2 norm between the true and predicted causal graphs.
        Also calculates the L2 norm on just the states that had non-trivial causal graphs.
        """
        mask = torch.ones((self.env.num_obj, self.env.num_obj), dtype=torch.bool).to(
            DEVICE
        )
        mask = mask ^ torch.eye(self.env.num_obj, dtype=torch.bool).to(DEVICE)
        collisions_tensor = torch.cat(true_collisions)

        masked_collisions = torch.masked_select(collisions_tensor, mask).view(
            collisions_tensor.shape[0], -1
        )
        had_collision_index = torch.nonzero(masked_collisions, as_tuple=False)[
            :, 0
        ].unique()
        true_had = masked_collisions[had_collision_index]
        pred_had = pred_collisions[had_collision_index]

        coll_losses = torch.norm(true_had - pred_had, dim=1)
        full_coll_losses = torch.norm(masked_collisions - pred_collisions, dim=1)
        return coll_losses, full_coll_losses

    def test(self) -> Tuple[float, float]:
        """Test a trained model.

        Returns:
            Average state prediction and graph prediction loss.
        """
        test_size = 1000
        dataset = SimpleDataset(self.env, test_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.graph_predictor.eval()
        self.state_predictor.eval()

        losses = torch.zeros(test_size)
        pred_collisions = torch.zeros((test_size, self.graph_size)).to(DEVICE)
        true_collisions: List[torch.Tensor] = []

        for i, (
            node_state_batch,
            true_state_batch,
            true_graph_batch,
            true_indices,
        ) in enumerate(dataloader):
            mini_batch_size = len(node_state_batch)
            storage_index = i * self.batch_size

            # The raw predictions for the upper triangle of the adjacency matrix.
            pred_adj_matrix_probabilities = self.graph_predictor(
                node_state_batch.view(mini_batch_size, -1)
            )
            if not self.variational:
                pred_adj_matrix = pred_adj_matrix_probabilities
            else:
                dist = distributions.Bernoulli(probs=pred_adj_matrix_probabilities)
                pred_adj_matrix = dist.sample()

            state_pred = self.state_predictor(node_state_batch, pred_adj_matrix)

            # Calculate losses and backprop

            # MSE Loss
            loss = functional.mse_loss(
                state_pred, true_state_batch, reduction="none"
            ).view(mini_batch_size, -1)
            batch_loss = loss.sum(dim=1) / loss.shape[1]
            mean_loss = batch_loss.mean()

            # Store everything
            pred_collisions[
                storage_index : storage_index + mini_batch_size
            ] = pred_adj_matrix_probabilities
            true_collisions.append(true_graph_batch)
            batch_norm = torch.norm(
                (true_state_batch - state_pred).view(mini_batch_size, -1), dim=1
            )
            losses[
                storage_index : storage_index + mini_batch_size
            ] = batch_loss.detach()
        collision_losses, full_coll_losses = self.collision_losses(
            true_collisions, pred_collisions
        )

        return losses.mean().item(), full_coll_losses.mean().item()

    def save(self):
        torch.save(
            self.graph_predictor.state_dict(),
            str(MODELS / "{}_graph.tch".format(self.display_name)),
        )
        torch.save(
            self.state_predictor.state_dict(),
            str(MODELS / "{}_state.tch".format(self.display_name)),
        )

    def load(self):
        self.graph_predictor.load_state_dict(
            torch.load(str(MODELS / "{}_graph.tch".format(self.display_name)))
        )
        self.state_predictor.load_state_dict(
            torch.load(str(MODELS / "{}_state.tch".format(self.display_name)))
        )


class Iterated(FullArch):
    def __init__(self, *args, g_batches: int = 128, s_batches: int = 128, **kwargs):
        kwargs["reg_param"] = 0.0
        super().__init__(*args, **kwargs)
        self.g_batches = g_batches
        self.s_batches = s_batches
        self.use_g = True
        self.batch = 0

    def backprop(self, g_opt: Optimizer, s_opt: Optimizer):
        self.batch += 1

        # TODO: Instead of using separate optimizers, alternately freeze models
        # This integrates better with a FullModel setup.
        if self.use_g:
            g_opt.step()
            if self.batch == self.g_batches:
                self.batch = 0
                self.use_g = False
        else:
            s_opt.step()
            if self.batch == self.s_batches:
                self.batch = 0
                self.use_g = True
        g_opt.zero_grad()
        s_opt.zero_grad()


class PreBuffer(FullArch):
    """An experiment where we first generate a buffer of high-error states, then train with those"""

    name = "prebuffer"

    def __init__(self, *args, buffer_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # self.buffer_prob = 0.4
        self.buffer_start = 250
        self.generating_arch = FullArch(
            self.env, min(self.steps * 12, 12000), self.lr, **kwargs
        )

    def get_data(self) -> Tuple[StateAndGraph, DataLoader]:
        "Get a dataset and dataloader for the current environment"
        self.save_models = True

        self.dataset: StateAndGraph = StoredBuffer(
            self.env, self.buffer_prob, self.generating_arch
        )
        # Reset steps to match the actual number of steps.
        self.steps = len(self.dataset)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        return self.dataset, self.dataloader


# Death before mixins
class IterBuffer(PreBuffer):
    def __init__(
        self, *args, g_batches: int = 128, s_batches: int = 128, reg_param=0.0, **kwargs
    ):
        super().__init__(*args, reg_param=reg_param, **kwargs)
        self.g_batches = g_batches
        self.s_batches = s_batches
        self.use_g = True
        self.batch = 0

    def backprop(self, g_opt: Optimizer, s_opt: Optimizer):
        self.batch += 1

        # TODO: Instead of using separate optimizers, alternately freeze models
        # This integrates better with a FullModel setup.
        if self.use_g:
            g_opt.step()
            if self.batch == self.g_batches:
                self.batch = 0
                self.use_g = False
        else:
            s_opt.step()
            if self.batch == self.s_batches:
                self.batch = 0
                self.use_g = True
        g_opt.zero_grad()
        s_opt.zero_grad()


class PerfectGraph(Experiment):
    """Perform a test using the true collision graphs and a state predictor."""

    name = "just_state"

    def __init__(
        self,
        env,
        steps: int,
        lr: float,
        gen_new: bool = True,
        buffer_prob: float = 0.0,
        variational: bool = False,
        batch_size: int = 32,
        msg_widths: Tuple[int, ...] = (32, 32),
        final_widths: Tuple[int, ...] = (32, 32),
    ):
        super().__init__(env, steps, lr, gen_new=gen_new)
        self.buffer_prob = buffer_prob
        self.variational = variational
        self.batch_size = batch_size
        if variational:
            self.state_predictor = DiscretePredictor(
                self.env.num_obj, self.env.obj_dim
            ).to(DEVICE)
        else:
            self.state_predictor = WeightedPredictor(
                self.env.num_obj,
                self.env.obj_dim,
                msg_widths=msg_widths,
                final_widths=final_widths,
            ).to(DEVICE)

    @property
    def display_name(self):
        extra = ""
        if self.variational:
            extra += "_variational"
        return "{}_{}".format(self.name, self.steps) + extra

    def get_data(self) -> Tuple[SimpleDataset, DataLoader]:
        if self.buffer_prob > 0.0:
            dataset: SimpleDataset = BufferedState(
                self.env, self.steps, self.buffer_prob
            )
        else:
            dataset = SimpleDataset(self.env, self.steps)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return self.dataset, self.dataloader

    def run(self):
        s_opt = optim.Adam(self.state_predictor.parameters(), lr=self.lr)

        dataset, dataloader = self.get_data()

        losses = torch.zeros(self.steps)

        mask = torch.ones((self.env.num_obj, self.env.num_obj), dtype=torch.bool).to(
            DEVICE
        )
        mask = mask ^ torch.eye(self.env.num_obj, dtype=torch.bool).to(DEVICE)

        for i, (node_state, true_state, true_graph, true_index) in enumerate(
            dataloader
        ):
            batch_size = node_state.shape[0]
            # Remove diagonal
            no_diag = torch.masked_select(true_graph, mask).view(batch_size, -1)
            state_pred = self.state_predictor(node_state, no_diag)

            # Backprop
            s_opt.zero_grad()

            # MSE Loss
            loss = functional.mse_loss(state_pred, true_state, reduction="none").view(
                batch_size, -1
            )
            batch_loss = loss.sum(dim=1) / loss.shape[1]

            mean_loss = batch_loss.mean()
            if i > 50 and mean_loss > 2 * losses[i - 50 : i].mean():
                dataset.update(true_index)

            mean_loss.backward()

            s_opt.step()

            losses[i * batch_size : (i + 1) * batch_size] = batch_loss.detach()

        return losses.detach()

    def test(self) -> Tuple[float, float]:
        """Test a trained model.

        Returns:
            Average state prediction and graph prediction loss.
        """
        test_size = 1000
        dataset = SimpleDataset(self.env, test_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.state_predictor.eval()

        losses = torch.zeros(test_size)

        mask = torch.ones((self.env.num_obj, self.env.num_obj), dtype=torch.bool).to(
            DEVICE
        )
        mask = mask ^ torch.eye(self.env.num_obj, dtype=torch.bool).to(DEVICE)

        for i, (node_state, true_state, true_graph, true_index) in enumerate(
            dataloader
        ):
            batch_size = node_state.shape[0]
            # Remove diagonal
            no_diag = torch.masked_select(true_graph, mask).view(batch_size, -1)
            state_pred = self.state_predictor(node_state, no_diag)

            # MSE Loss
            loss = functional.mse_loss(state_pred, true_state, reduction="none").view(
                batch_size, -1
            )
            batch_loss = loss.sum(dim=1) / loss.shape[1]

            losses[i * batch_size : (i + 1) * batch_size] = batch_loss.detach()

        return losses.mean().item(), 0.0

    def load_or_gen(self):
        # try:
        #     self.load()
        # except FileNotFoundError:
        self.run()
        # self.save()

    def save(self):
        torch.save(
            self.state_predictor.state_dict(),
            str(MODELS / "{}_state.tch".format(self.display_name)),
        )

    def load(self):
        self.state_predictor.load_state_dict(
            torch.load(str(MODELS / "{}_state.tch".format(self.display_name)))
        )


class PerfectState(FullArch):
    """Train the graph predictor with a pretrained state predictor."""

    name = "perfect_state"

    def __init__(self, env, steps: int, lr: float, **kwargs):
        super().__init__(
            env, steps, lr, msg_widths=(8, 8), final_widths=(8, 8), **kwargs
        )

        self.just_state = PerfectGraph(
            env, steps * 5, 0.1, buffer_prob=0.75, variational=True
        )
        self.iteration_steps = self.steps * self.epochs
        self.gen_new_model = False
        self.just_state.state_predictor = self.state_predictor

    def get_data(self):
        if self.gen_new_model:
            self.just_state.run()
        else:
            self.just_state.load_or_gen()
        self.state_predictor = self.just_state.state_predictor
        return super().get_data()


class FullModel(Experiment):
    """The full architecture experiment.

    Attributes:
        env (Environment):                  The environment to run on.
        graph_predictor (CDAGPredictor):    A neural network whose output is a distribution on Bipartite(k, k), where k is the number of
                                                objects in the environment.
        state_predictor (nn.Module):        A neural network from (current_state, Bipartite(k, k)) -> next_state.
        variational (bool):                 Whether or not to use the weighted or the sampling architecture.
        asymmetric (bool):                  Whether or not the model should predict asymmetric causal graphs.
        convolutional (bool):               Whether or not the graph predictor should be convolutional or fully-connected
        local (bool):                       Whether or not to force the graph predictor to only predict local causal relations.
        epochs (int):                       Number of epochs to run for.
        steps (int):                        Number of steps in an epoch.
        batch_size (int):                   Size of an individual batch.
        lr (float):                         The initial learning rate for Adam.
        reg_param (float):                  L1 regularization parameter on the causal graph prediction.
    """

    name = "full"

    def __init__(
        self,
        env,
        steps: int,
        lr: float,
        epochs: int = 1,
        gen_new: bool = True,
        reg_param: float = 1.0,
        lin_widths: Tuple[int, ...] = (32, 32),
        msg_widths: Tuple[int, ...] = (32, 32),
        final_widths: Tuple[int, ...] = (32, 32),
        asymmetric: bool = False,
        convolutional: bool = False,
        local: bool = False,
        iteration_steps: int = 0,
        bad_state_weight: float = 1.0,
        variational: bool = False,
        buffer_prob: float = 0.0,
        max_distance: float = 30.0,
        batch_size: int = 32,
        save_model: bool = False,
    ):
        super().__init__(env, steps, lr, gen_new=gen_new)

        self.save_model = save_model

        # Training parameters
        self.epochs = epochs
        self.reg_param = reg_param
        self.batch_size = batch_size
        self.buffer_prob = buffer_prob
        self.iteration_steps = iteration_steps
        self.bad_state_weight = bad_state_weight

        # Model parameters
        self.asymmetric = asymmetric
        self.convolutional = convolutional
        self.variational = variational
        self.local = local
        self.max_distance = max_distance

        self.model = GraphAndMessages(self.env.num_obj, self.env.obj_dim).to(DEVICE)

    @property
    def graph_size(self):
        return self.env.num_obj * (self.env.num_obj - 1)

    @property
    def display_name(self):
        extra = ""
        if self.asymmetric:
            extra += "_asymmetric"
        if self.variational:
            extra += "_variational"
        if self.bad_state_weight != 1.0:
            extra += "_overweight({})".format(self.bad_state_weight)
        if self.iteration_steps > 0:
            extra += "_iteration({})".format(self.iteration_steps)
        if self.local:
            extra += "_local"
        if self.convolutional:
            extra += "_wconvs"
        if self.buffer_prob > 0.0:
            extra += "_buffer({})".format(self.buffer_prob)
        return "{}_{}".format(self.name, self.steps) + extra

    def get_data(self) -> Tuple[StateAndGraph, DataLoader]:
        "Get a dataset and dataloader for the current environment"
        if self.buffer_prob > 0.0:
            dataset: StateAndGraph = BufferedState(
                self.env, self.steps, self.buffer_prob
            )
        else:
            dataset = SimpleDataset(self.env, self.steps)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return self.dataset, self.dataloader

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the entire training loop

        Returns:
            Losses and collision losses
        """
        # Setup
        dataset, dataloader = self.get_data()

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        opt.zero_grad()

        losses = torch.zeros(self.epochs * self.steps)
        pred_collisions = torch.zeros((self.epochs * self.steps, self.graph_size)).to(
            DEVICE
        )

        # The true collisions at each step
        true_collisions: List[torch.Tensor] = []

        # The gradients of the graph predictor at each step
        graph_gradients: List[List[float]] = []

        for e in range(self.epochs):
            print("Epoch: {}".format(e))

            for i, (
                node_state_batch,
                true_state_batch,
                true_graph_batch,
                true_indices,
            ) in enumerate(dataloader):
                mini_batch_size = len(node_state_batch)
                storage_index = e * self.steps + i * self.batch_size

                # Next state prediction
                pred_adj_matrix_probabilities, state_pred = self.model(node_state_batch)

                ## Calculate losses and backprop

                # MSE Loss
                loss = functional.mse_loss(
                    state_pred, true_state_batch, reduction="none"
                ).view(mini_batch_size, -1)
                batch_loss = loss.sum(dim=1) / loss.shape[1]

                # Handle bad states
                if storage_index > self.batch_size * 5:
                    max_loss = (
                        2.0
                        * losses[
                            storage_index - self.batch_size // 2 : storage_index
                        ].mean()
                    )
                    filt = batch_loss > max_loss

                    # Store bad state indices
                    bad_indices = true_indices[filt.nonzero(as_tuple=False).view(-1)]
                    dataset.update(bad_indices)

                    # Overweight bad states
                    batch_loss[filt] *= self.bad_state_weight

                mean_loss = batch_loss.mean()

                # L1 regularization for the graph predictor
                if self.reg_param > 0:
                    l1_reg = self.reg_param * torch.norm(
                        pred_adj_matrix_probabilities, p=1
                    )
                    combined_loss = l1_reg + mean_loss
                else:
                    combined_loss = mean_loss

                combined_loss.backward(retain_graph=True)

                opt.step()
                opt.zero_grad()

                # Store everything
                pred_collisions[
                    storage_index : storage_index + mini_batch_size
                ] = pred_adj_matrix_probabilities
                true_collisions.append(true_graph_batch)
                losses[
                    storage_index : storage_index + mini_batch_size
                ] = (
                    batch_loss.detach()
                )  # / true_state_batch.view(mini_batch_size, -1).norm(dim=1)
                self.writer.add_scalar(
                    "State L2 Loss", combined_loss.item(), storage_index
                )

        if self.save_model:
            self.save()

        # Compute collision losses
        collision_losses, full_coll_losses = self.collision_losses(
            true_collisions, pred_collisions
        )
        for i, cl in enumerate(collision_losses):
            self.writer.add_scalar("Collision L2 Loss", cl, i)

        return losses, full_coll_losses.detach()

    def collision_losses(
        self, true_collisions: List[torch.Tensor], pred_collisions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the L2 norm between the true and predicted causal graphs.
        Also calculates the L2 norm on just the states that had non-trivial causal graphs.
        """
        mask = torch.ones((self.env.num_obj, self.env.num_obj), dtype=torch.bool).to(
            DEVICE
        )
        mask = mask ^ torch.eye(self.env.num_obj, dtype=torch.bool).to(DEVICE)
        collisions_tensor = torch.cat(true_collisions)

        masked_collisions = torch.masked_select(collisions_tensor, mask).view(
            collisions_tensor.shape[0], -1
        )
        had_collision_index = torch.nonzero(masked_collisions, as_tuple=False)[
            :, 0
        ].unique()
        true_had = masked_collisions[had_collision_index]
        pred_had = pred_collisions[had_collision_index]

        coll_losses = torch.norm(true_had - pred_had, dim=1)
        full_coll_losses = torch.norm(masked_collisions - pred_collisions, dim=1)
        return coll_losses, full_coll_losses

    def save(self):
        torch.save(
            self.graph_predictor.state_dict(),
            str(MODELS / "{}_graph.tch".format(self.display_name)),
        )
        torch.save(
            self.state_predictor.state_dict(),
            str(MODELS / "{}_state.tch".format(self.display_name)),
        )

    def load(self):
        self.graph_predictor.load_state_dict(
            torch.load(str(MODELS / "{}_graph.tch".format(self.display_name)))
        )
        self.state_predictor.load_state_dict(
            torch.load(str(MODELS / "{}_state.tch".format(self.display_name)))
        )


EXPERIMENT_MAP = {
    "just_state": PerfectGraph,
    "naive_model": NaiveModel,
    "full_arch": FullArch,
    "perfect_state": PerfectState,
    "prebuffer": PreBuffer,
    "e2e": FullModel,
}


def full_arch(
    exp_name: str = "full_arch",
    env_name: str = "bouncing_balls",
    steps: int = 3000,
    epochs: int = 1,
    lr: float = 0.1,
    variational: bool = True,
    convolutional: bool = True,
    local: bool = False,
    asymmetric: bool = False,
    buffer_prob: float = 0.0,
    iteration_steps: int = 0,
    env_kwargs={"num_obj": 5, "width": 100},
):
    """Run a full architecture experiment."""
    env_class = ENVIRONMENT_MAP[env_name]
    exp_class = EXPERIMENT_MAP[exp_name]
    env = env_class(**env_kwargs)
    exp = exp_class(
        env,
        steps,
        lr,
        reg_param=0.1,
        asymmetric=asymmetric,
        variational=variational,
        convolutional=convolutional,
        local=local,
        epochs=epochs,
        buffer_prob=buffer_prob,
        iteration_steps=iteration_steps,
    )

    losses, coll_losses = exp.run()
    plot_exploration(
        coll_losses.cpu(), exp.display_name + "_coll_losses", env.name, save=True
    )
    plot_exploration(losses, exp.display_name, env.name, save=True)
    print(exp.test())
    # plot_grad(graph_grads, 'tmp', exp.name + '_graph_grads')


def w_v(steps: int = 3000):
    """Weighted vs variational state predictor."""
    # env_sizes = (250, 750)
    # num_objs = (5, 25)
    folder = Path("weight_vs_var")

    for env_size, num_obj in ((100, 5), (250, 15), (500, 25)):
        env = HardSpheres(num_obj=num_obj, width=env_size)

        for variational in (True, False):
            exp = FullArch(
                env, steps=steps, lr=0.1, variational=variational, convolutional=False
            )
            losses, coll_losses = exp.run()

            exp_name = "{}_{}_{}".format(steps, num_obj, env_size)
            exp_name = ("var_" if variational else "full_") + exp_name
            plot_coll_loss(coll_losses.cpu(), folder, exp_name)
            plot_state_loss(losses, folder, exp_name)


def naive_pred(steps: int = 3000, lr: float = 0.01, epochs: int = 10):
    """Run a non-causal model to predict states"""
    env = HardSpheres(num_obj=15, width=250)
    folder = "naive"
    exp = NaiveModel(env, steps, lr, epochs=epochs)

    losses, _ = exp.run()
    plot_exploration(losses.cpu(), exp.display_name, env.name, save=True)


def train_buffer():
    """Run experiments with buffered bad states."""
    folder = Path("train_buffer/")
    env = HardSpheres(num_obj=5, width=100)

    for buffer_prob in (0.1, 0.25, 0.5, 0.75, 1.0):
        exp = FullArch(env, steps=10000, lr=0.1, buffer_prob=buffer_prob)
        losses, coll_losses = exp.run()

        exp_name = "buffer({})".format(buffer_prob)
        plot_coll_loss(coll_losses.cpu(), folder, exp_name)
        plot_state_loss(losses, folder, exp_name)


def train_prebuffer():
    """Run experiments with buffered bad states."""
    folder = "train_prebuffer/"
    env = HardSpheres(num_obj=5, width=100)
    steps = 5000
    lr = 0.1

    for buffer_prob in (0.25, 0.5, 0.75):
        exp = PreBuffer(env, steps, lr, buffer_prob=buffer_prob)
        losses, coll_losses = exp.run()
        exp_name = "prebuffer({})".format(buffer_prob)
        plot_coll_loss(coll_losses.cpu(), folder, exp_name)
        plot_state_loss(losses, folder, exp_name)


def train_iter():
    """Run experiments with the iterated training algorithm."""
    folder = "train_iter/"
    env = HardSpheres(num_obj=5, width=100)
    steps = 3000
    lr = 0.1
    # g_batches = 15
    # s_batches = 1
    for g_batches, s_batches in ((15, 1), (30, 1), (10, 5)):
        exp = Iterated(
            env, steps, lr, epochs=5, g_batches=g_batches, s_batches=s_batches
        )
        losses, coll_losses = exp.run()
        exp_name = "iter({}, {})".format(g_batches, s_batches)
        plot_coll_loss(coll_losses.cpu(), folder, exp_name)
        plot_state_loss(losses, folder, exp_name)


def train_iterbuf():
    """Run experiments with iterated learning and a preconstructed buffer"""
    folder = "train_iterbuf/"
    env = HardSpheres(num_obj=5, width=100)
    steps = 3000
    lr = 0.1
    for g_batches, s_batches in ((15, 1), (30, 1), (10, 5)):
        exp = IterBuffer(
            env,
            steps,
            lr,
            epochs=3,
            variational=True,
            g_batches=g_batches,
            s_batches=s_batches,
        )
        losses, coll_losses, graph_grads = exp.run()
        exp_name = "iterbuf({}, {})".format(g_batches, s_batches)
        plot_coll_loss(coll_losses.cpu(), folder, exp_name)
        plot_state_loss(losses, folder, exp_name)
        plot_grad(graph_grads, folder, exp_name + "_graph_grads")


def convolutional(steps: int = 3000):
    """Assess convolutional graph predictor."""
    folder = Path("convolutional")

    for variational in (True, False):
        for env_size, num_obj in ((250, 25), (500, 50)):
            env = HardSpheres(num_obj=num_obj, width=env_size)

            for convolutional in (True, False):
                exp = FullArch(
                    env,
                    steps=steps,
                    lr=0.1,
                    convolutional=convolutional,
                    variational=variational,
                )
                losses, coll_losses = exp.run()

                exp_name = "{}_{}_{}".format(num_obj, env_size, variational)
                exp_name = ("conv_" if convolutional else "full_") + exp_name
                plot_coll_loss(coll_losses.cpu(), folder, exp_name)
                plot_state_loss(losses, folder, exp_name)


def local_section(steps: int = 3000):
    folder = Path("local_section")

    for env_size, num_obj in ((250, 25), (500, 50)):
        env = HardSpheres(num_obj=num_obj, width=env_size)

        for local in (True, False):
            exp = FullArch(env, steps=steps, lr=0.1, local=local, convolutional=False)
            losses, coll_losses = exp.run()

            exp_name = "({}, {})".format(num_obj, env_size)
            exp_name = ("local" if local else "full") + exp_name
            plot_coll_loss(coll_losses.cpu(), folder, exp_name)
            plot_state_loss(losses, folder, exp_name)


def action_reaction():
    """Run experiments with and without the action-reaction assumption."""
    folder = "action_reaction/"
    env = HardSpheres(num_obj=15, width=250)

    for asymmetric in (True, False):
        exp = FullArch(env, steps=3000, lr=0.1, asymmetric=asymmetric)
        losses, coll_losses = exp.run()
        exp_name = "asymm" if asymmetric else "symm"
        plot_coll_loss(coll_losses.cpu(), folder, exp_name)
        plot_state_loss(losses, folder, exp_name)


def uniform_objs():
    """Run experiments with and without the uniform objects assumption"""
    env = HardSpheres(num_obj=25, width=300)

    for convolutional in (True, False):
        exp = FullArch(env, steps=1000, lr=0.1, convolutional=convolutional)
        losses, coll_losses = exp.run()
        plot_exploration(
            coll_losses.detach().cpu().numpy(),
            "unif_objs/" + exp.display_name + "_coll_losses",
            env.name,
            save=True,
            add_title=False,
        )
        plot_exploration(
            losses,
            "unif_objs/" + exp.display_name,
            env.name,
            save=True,
            add_title=False,
        )


def train_overweight():
    """Run experiments with buffered bad states."""
    folder = Path("train_overweight/")
    env = HardSpheres(num_obj=5, width=200)

    for weight in (1.0, 2.0, 10.0, 25.0):
        exp = FullArch(env, steps=7000, lr=0.1, bad_state_weight=weight)
        losses, coll_losses = exp.run()

        exp_name = "overweight({})".format(weight)
        plot_display(coll_losses.detach().cpu().numpy(), folder, "coll_" + exp_name, 20)
        plot_display(losses, folder, exp_name, 3000)


def perfect_graph(steps: int = 3000):
    """Run an experiment with the true collision graph given."""
    folder = "perfect_graph/"
    env = HardSpheres(num_obj=15, width=250)

    for lr in (0.01, 0.05, 0.1, 0.5):
        exp = PerfectGraph(env, steps, lr)
        losses = exp.run()
        print(exp.test())
        exp_name = "perfect_graph({})".format(lr)
        plot_state_loss(losses, folder, exp_name)


def perfect_state():
    folder = "perfect_state/"
    env = HardSpheres(num_obj=5, width=200)
    steps = 3000

    for lr in (0.01, 0.05, 0.1, 0.5):
        exp = PerfectState(env, steps, lr)
        losses, coll_losses = exp.run()
        exp_name = "perfect_state({})".format(lr)
        plot_coll_loss(coll_losses.cpu(), folder, "coll_" + exp_name)
        plot_state_loss(losses, folder, exp_name)


def multi_typed():
    folder = "multi_typed/"
    for num_obj, env_size in ((5, 100), (15, 250), (25, 500)):
        for env_class in (MultiTyped, WithTypes):
            env = env_class(num_obj=num_obj, width=env_size)

            # for variational in (True, False):
            # for local in (True, False):
            exp = FullArch(env, steps=5000, lr=0.1, reg_param=1)
            losses, coll_losses = exp.run()
            exp_name = "{}_{}".format(env.name, num_obj)
            print(exp.test())
            plot_coll_loss(coll_losses.cpu(), folder, exp_name)
            plot_state_loss(losses, folder, exp_name)


def mujoco(steps: int = 3000):
    folder = Path("mujoco")

    for env_name in ("Hopper-v2", "HalfCheetah-v2", "Ant-v2"):
        env = Mujoco(env_name=env_name)

        exp = FullArch(env, steps=steps, lr=0.1, convolutional=False)
        losses, coll_losses = exp.run()

        exp_name = "{}_{}".format(env_name, steps)
        plot_display(losses, folder, exp_name, 10)


def shared(steps=1000):
    """Test the shared parameters network"""
    folder = Path("tmp")

    for env_size, num_obj in ((100, 5), (250, 15), (500, 25)):
        env = HardSpheres(num_obj=num_obj, width=env_size)

        exp = FullModel(env, steps=steps, lr=0.1)
        losses, coll_losses = exp.run()

        exp_name = "{}_{}_{}".format(steps, num_obj, env_size)
        exp_name = "shared_" + exp_name
        plot_coll_loss(coll_losses.cpu(), folder, exp_name)
        plot_state_loss(losses, folder, exp_name)


def plot_learning_comparison():
    train_losses = np.genfromtxt(
        str(PLOTS / "perfect_graph" / "learning_comp.csv"), delimiter=","
    )
    fig, ax = plt.subplots()
    x_axis = range(10, 100, 10)
    ax.plot(
        x_axis,
        train_losses[:, 0],
        linestyle="None",
        marker="o",
        markersize=3,
        label="With True Graph",
    )
    ax.plot(
        x_axis,
        train_losses[:, 1],
        linestyle="None",
        marker="o",
        markersize=3,
        label="Sampling",
    )
    ax.plot(
        x_axis,
        train_losses[:, 2],
        linestyle="None",
        marker="o",
        markersize=3,
        label="Weighted",
    )

    ax.set_ylim([0, 20000])
    ax.set_xlim([0, 100])
    ax.set_xlabel("Batch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Learning Rate Comparison")
    ax.legend()

    plt.savefig((str(PLOTS / "perfect_graph" / "learning_comp.png")))


if __name__ == "__main__":
    Fire()
