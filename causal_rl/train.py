"""Training algorithms for local representations.

Algorithms implemented as classes since there are a variety of slight variations.
"""

import numpy as np
import torch

from fire import Fire
from itertools import product
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
from causal_rl.graph_predictor import FullyConnectedPredictor, ConvLinkPredictor, CDAGPredictor
from causal_rl.state_predictor import DiscretePredictor, WeightedPredictor
from causal_rl.model import NaivePredictor, CausalPredictor, LocalRepresentation
from causal_rl.utils import plot_exploration, plot_coll_loss, plot_state_loss, load_sim, store_sim


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    name = 'full'

    def __init__(self, model: LocalRepresentation, steps: int, lr: float, epochs: int=1, gen_new: bool=True, reg_param: float=1.0,
                iteration_steps: int=0,
                bad_state_weight: float=1.0,
                buffer_prob: float=0.0,
                batch_size: int=32,
                save_model: bool=False):
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

    def run(self, env: CausalEnv, dataset, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the entire training loop

        Returns:
            Losses and collision losses
        """
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        opt.zero_grad()

        losses = torch.zeros(self.epochs * self.steps)
        pred_collisions = torch.zeros((self.epochs * self.steps, self.graph_size)).to(DEVICE)

        # The true collisions at each step
        true_collisions: List[torch.Tensor] = []

        # The gradients of the graph predictor at each step
        # graph_gradients: List[List[float]] = []

        for e in range(self.epochs):
            print("Epoch: {}".format(e))

            for i, (node_state_batch, true_state_batch, true_graph_batch, true_indices) in enumerate(dataloader):
                mini_batch_size = len(node_state_batch)
                storage_index = e * self.steps + i * self.batch_size

                # TODO: Move state and graph prediction into FullModel class.
                # The raw predictions for the upper triangle of the adjacency matrix.
                pred_adj_matrix_probabilities, pred_state = self.model(node_state_batch)

                ## Calculate losses and backprop

                # MSE Loss
                loss = functional.mse_loss(pred_state, true_state_batch, reduction='none').view(mini_batch_size, -1)
                batch_loss = loss.sum(dim=1) / loss.shape[1]

                # Handle bad states
                if storage_index > self.batch_size * 5:
                    max_loss = 2.0 * losses[storage_index - self.batch_size // 2: storage_index].mean()
                    filt = batch_loss > max_loss

                    # Store bad state indices
                    bad_indices = true_indices[filt.nonzero(as_tuple=False).view(-1)]
                    dataset.update(bad_indices)

                    # Overweight bad states
                    batch_loss[filt] *= self.bad_state_weight

                mean_loss = batch_loss.mean()


                # L1 regularization for the graph predictor
                if self.reg_param > 0:
                    l1_reg = self.reg_param * torch.norm(pred_adj_matrix_probabilities, p=1)
                    combined_loss = l1_reg + mean_loss
                else:
                    combined_loss = mean_loss

                combined_loss.backward(retain_graph=True)

                # The loss for the graph predictor
                # Only defined when we use the variational architecture.
                # if self.variational:
                #     graph_loss = dist.log_prob(pred_adj_matrix) * combined_loss.detach()
                #     graph_loss.mean().backward()

                opt.step()
                opt.zero_grad()

                # Store everything
                pred_collisions[storage_index: storage_index + mini_batch_size] = pred_adj_matrix_probabilities
                true_collisions.append(true_graph_batch)
                batch_norm = torch.norm((true_state_batch - pred_state).view(mini_batch_size, -1), dim=1)
                losses[storage_index: storage_index + mini_batch_size] = batch_loss.detach() #/ true_state_batch.view(mini_batch_size, -1).norm(dim=1)
                # self.writer.add_scalar('State L2 Loss', combined_loss.item(), storage_index)

        # Compute collision losses
        collision_losses, full_coll_losses = self.collision_losses(true_collisions, pred_collisions)

        return losses, full_coll_losses.detach()

    def collision_losses(self, true_collisions: List[torch.Tensor], pred_collisions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the L2 norm between the true and predicted causal graphs.
        Also calculates the L2 norm on just the states that had non-trivial causal graphs.
        """
        num_obj = pred_collisions.shape[1]
        mask = torch.ones((num_obj, num_obj), dtype=torch.bool).to(DEVICE)
        mask = mask ^ torch.eye(num_obj, dtype=torch.bool).to(DEVICE)
        collisions_tensor = torch.cat(true_collisions)

        masked_collisions = torch.masked_select(collisions_tensor, mask).view(collisions_tensor.shape[0], -1)
        had_collision_index = torch.nonzero(masked_collisions, as_tuple=False)[:, 0].unique()
        true_had = masked_collisions[had_collision_index]
        pred_had = pred_collisions[had_collision_index]

        coll_losses = torch.norm(true_had - pred_had, dim=1)
        full_coll_losses = torch.norm(masked_collisions - pred_collisions, dim=1)
        return coll_losses, full_coll_losses

    # def save(self):
    #     torch.save(self.graph_predictor.state_dict(), str(MODELS / '{}_graph.tch'.format(self.display_name)))
    #     torch.save(self.state_predictor.state_dict(), str(MODELS / '{}_state.tch'.format(self.display_name)))

    # def load(self):
    #     self.graph_predictor.load_state_dict(torch.load(str(MODELS / '{}_graph.tch'.format(self.display_name))))
    #     self.state_predictor.load_state_dict(torch.load(str(MODELS / '{}_state.tch'.format(self.display_name))))


