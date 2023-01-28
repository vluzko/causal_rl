"""Training algorithms for local representations.

Algorithms implemented as classes since there are a variety of slight variations.
"""

import numpy as np
import torch

from itertools import product
from matplotlib import pyplot as plt
from pathlib import Path
from random import sample
from torch import optim, distributions
from torch.optim.optimizer import Optimizer
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

from causal_rl.environments import CausalEnv
from causal_rl.model import LocalRepresentation
from causal_rl.tracker import Tracker


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TrainNormal:
    """The full architecture experiment.

    Attributes:
        model (LocalRepresentation):        The model being trained.
        epochs (int):                       Number of epochs to run for.
        steps (int):                        Number of steps in an epoch.
        batch_size (int):                   Size of an individual batch.
        lr (float):                         The initial learning rate for Adam.
        reg_param (float):                  L1 regularization parameter on the causal graph prediction.
    """

    name = "full"

    def __init__(
        self,
        model: LocalRepresentation,
        steps: int,
        lr: float,
        epochs: int = 1,
        gen_new: bool = True,
        reg_param: float = 1.0,
        iteration_steps: int = 0,
        bad_state_weight: float = 1.0,
        buffer_prob: float = 0.0,
        batch_size: int = 32,
        save_model: bool = False,
    ):
        # super().__init__(env, steps, lr, gen_new=gen_new)
        self.steps = steps
        self.lr = lr
        self.gen_new = gen_new

        self.model = model
        self.save_model = save_model

        # Training parameters
        self.epochs = epochs
        self.reg_param = reg_param
        self.batch_size = batch_size
        self.buffer_prob = buffer_prob
        self.iteration_steps = iteration_steps
        self.bad_state_weight = bad_state_weight

    def run(self, dataset, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the entire training loop

        Returns:
            Losses and collision losses
        """
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        opt.zero_grad()

        losses = torch.zeros(self.epochs * self.steps)
        pred_collisions = []

        # The true collisions at each step
        true_collisions: List[torch.Tensor] = []

        # The gradients of the graph predictor at each step
        # graph_gradients: List[List[float]] = []

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
                pred_adj_matrix_probabilities, pred_state = self.model(node_state_batch)

                ## Calculate losses and backprop

                # MSE Loss
                loss = functional.mse_loss(
                    pred_state, true_state_batch, reduction="none"
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
                    l1_reg = self.reg_param * torch.norm(pred_adj_matrix_probabilities, p=1)  # type: ignore
                    combined_loss = l1_reg + mean_loss
                else:
                    combined_loss = mean_loss

                combined_loss.backward(retain_graph=True)

                opt.step()
                opt.zero_grad()

                # Store everything
                # pred_collisions[storage_index : storage_index + mini_batch_size] = pred_adj_matrix_probabilities
                pred_collisions.append(pred_adj_matrix_probabilities)
                true_collisions.append(true_graph_batch)
                # batch_norm = torch.norm((true_state_batch - pred_state).view(mini_batch_size, -1), dim=1)
                losses[
                    storage_index : storage_index + mini_batch_size
                ] = (
                    batch_loss.detach()
                )  # / true_state_batch.view(mini_batch_size, -1).norm(dim=1)

        # Compute collision losses
        collision_losses, full_coll_losses = self.collision_losses(
            true_collisions, torch.tensor(pred_collisions)
        )

        return losses, full_coll_losses.detach()

    def collision_losses(
        self, true_collisions: List[torch.Tensor], pred_collisions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the L2 norm between the true and predicted causal graphs.
        Also calculates the L2 norm on just the states that had non-trivial causal graphs.
        """
        num_obj = pred_collisions.shape[1]
        mask = torch.ones((num_obj, num_obj), dtype=torch.bool).to(DEVICE)
        mask = mask ^ torch.eye(num_obj, dtype=torch.bool).to(DEVICE)
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


class TrainGumbel:
    """Train a gumbel-softmax based VAE

    Attributes:
        model: The encoder-decoder
        categorical_dim: The number of edge categories. Hard coded to 2 for now.
        lr: Adam learning rate
        anneal_rate: Rate we anneal the temperature of the softmax at
    """

    def __init__(
        self,
        model: LocalRepresentation,
        lr: float = 1e-3,
        anneal_rate: float = 3e-5,
        start_temp: float = 1.0,
        temp_min: float = 0.5,
        hard: bool = False,
        save_model: bool = False,
    ):
        self.model = model
        self.categorical_dim = 2

        # Training parameters
        self.lr = lr
        self.start_temp = start_temp
        self.temp_min = temp_min
        self.anneal_rate = anneal_rate
        self.hard = hard

    def train(self, train_loader: DataLoader, tracker: Tracker):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        temp = self.start_temp
        for batch_idx, (feature, target, edge) in enumerate(train_loader):
            prediction, qy, graph_prediction = self.model(feature)
            loss = self.loss_function(prediction, target, qy)
            graph_loss = functional.binary_cross_entropy(graph_prediction, edge)
            loss.backward()
            optimizer.step()

            # Anneal the temperature
            if batch_idx % 100 == 1:
                temp = np.maximum(
                    temp * np.exp(-self.anneal_rate * batch_idx), self.temp_min
                )

            optimizer.zero_grad()
            tracker.add_scalar("Graph loss", graph_loss.mean().item(), batch_idx)
            tracker.add_scalar("Loss", loss.mean().item(), batch_idx)

    def loss_function(
        self, prediction: torch.Tensor, target: torch.Tensor, qy: torch.Tensor
    ) -> torch.Tensor:
        """Calculate VAE loss: main loss and a KL term."""
        main_loss = functional.mse_loss(prediction, target.view(target.shape[0], -1))
        log_ratio = torch.log(qy * self.categorical_dim + 1e-20)
        kl_div = torch.sum(qy * log_ratio, dim=-1).mean()
        return main_loss + kl_div

    def validate(self, valid_loader: DataLoader, optimizer):
        raise NotImplementedError

    #     self.model.eval()
    #     for batch_idx, (data, _) in enumerate(valid_loader):
    #         optimizer.zero_grad()
    #         prediction, qy = self.model(data)
    #         loss = self.loss_function(prediction, data, qy)
    # if batch_idx % 100 == 1:
    #     temp = np.maximum(temp * np.exp(-self.anneal_rate * batch_idx), self.temp_min)
