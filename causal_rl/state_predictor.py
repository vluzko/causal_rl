from einops import rearrange
import torch
from torch import nn
from torch.nn import Sequential, Linear, LeakyReLU, functional as F

from typing import Tuple, List

from causal_rl import utils


class DiscretePredictor(nn.Module):
    """Make predictions using a discrete graph on nodes."""

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        msg_widths: Tuple[int, ...] = (16,),
        final_widths: Tuple[int, ...] = (16,),
    ):
        super().__init__()
        self.num_obj = num_obj
        self.obj_dim = obj_dim
        self.size = num_obj

        # Set up the message passing neural network
        msg_layers: List[nn.Module] = [nn.Linear(self.obj_dim * 2, msg_widths[0])]

        for prev_width, width in zip(msg_widths[:-1], msg_widths[1:]):
            msg_layers.append(nn.BatchNorm1d(prev_width))
            msg_layers.append(nn.LeakyReLU())  # type: ignore
            msg_layers.append(nn.Linear(prev_width, width))
        msg_layers.append(nn.BatchNorm1d(msg_widths[-1]))
        msg_layers.append(nn.LeakyReLU())
        msg_layers.append(nn.Linear(msg_widths[-1], self.obj_dim))

        self.msg = nn.Sequential(*msg_layers)

        # Set up the final neural network
        final_layers: List[nn.Module] = [nn.Linear(self.obj_dim * 2, final_widths[0])]

        for prev_width, width in zip(final_widths[:-1], final_widths[1:]):
            final_layers.append(nn.BatchNorm1d(prev_width))
            final_layers.append(nn.LeakyReLU())  # type: ignore
            final_layers.append(nn.Linear(prev_width, width))
        final_layers.append(nn.BatchNorm1d(final_widths[-1]))
        final_layers.append(nn.LeakyReLU())
        final_layers.append(nn.Linear(final_widths[-1], self.obj_dim))

        self.final = nn.Sequential(*final_layers)

        self.source_indices, self.target_indices = utils.state_to_source_sink_indices(
            self.num_obj
        )

    def forward(self, state: torch.Tensor, edges: torch.Tensor):
        batch_size = state.shape[0]
        # The indices of the nonzero edges in the flattened node pairs matrix.
        edge_indices = torch.nonzero(edges, as_tuple=False)

        sources = state[:, self.source_indices]
        targets = state[:, self.target_indices]

        sources_with_edges = torch.mul(sources, edges.view(*edges.shape, 1))
        targets_with_edges = torch.mul(targets, edges.view(*edges.shape, 1))

        # Actual message passing

        need_message = torch.cat((sources_with_edges, targets_with_edges), dim=2)
        messages = self.msg(need_message.view(-1, self.obj_dim * 2))

        # Aggregation step
        reshaped_for_agg = messages.view(batch_size, state.shape[1], -1, state.shape[2])
        # Add the messages from each source node to a particular target node.
        # There is a separate channel for each component of the node vector
        aggregated = torch.sum(reshaped_for_agg, dim=2)

        # Final step
        state_and_messages = torch.cat((state, aggregated), dim=2)
        output = self.final(state_and_messages.view(-1, self.obj_dim * 2))

        return output.view(batch_size, -1, self.obj_dim)


class WeightedPredictor(nn.Module):
    """Predict the next state using the binomial distribution on each edge."""

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        msg_widths: Tuple[int, ...] = (32, 32),
        final_widths: Tuple[int, ...] = (32, 32),
    ):
        super().__init__()
        self.num_obj = num_obj
        self.obj_dim = obj_dim
        self.size = num_obj

        msg_layers: List[nn.Module] = [nn.Linear(self.obj_dim * 2, msg_widths[0])]

        for prev_width, width in zip(msg_widths[:-1], msg_widths[1:]):
            msg_layers.append(nn.BatchNorm1d(prev_width))
            msg_layers.append(LeakyReLU())  # type: ignore
            msg_layers.append(nn.Linear(prev_width, width))
        msg_layers.append(nn.BatchNorm1d(msg_widths[-1]))
        msg_layers.append(LeakyReLU())
        msg_layers.append(nn.Linear(msg_widths[-1], self.obj_dim))

        self.msg = nn.Sequential(*msg_layers)

        final_layers: List[nn.Module] = [nn.Linear(self.obj_dim * 2, msg_widths[0])]

        for prev_width, width in zip(final_widths[:-1], final_widths[1:]):
            final_layers.append(nn.BatchNorm1d(prev_width))
            final_layers.append(LeakyReLU())  # type: ignore
            final_layers.append(nn.Linear(prev_width, width))
        final_layers.append(nn.BatchNorm1d(final_widths[-1]))
        final_layers.append(LeakyReLU())
        final_layers.append(nn.Linear(final_widths[-1], self.obj_dim))

        self.final = nn.Sequential(*final_layers)
        self.target_indices, self.source_indices = utils.state_to_source_sink_indices(
            self.num_obj
        )

    def forward(self, state: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Pass messages.

        Args:
            state: The state batch. Should have shape (batch_size x num_obj x obj_dim)
            probs: The probabilities of edges between all objects. Should have shape (batch_size x (num_obj * (num_obj - 1)))
        """
        batch_size = state.shape[0]
        # The graph is fully connected, so we have a slightly weird architecture.
        target_node_states = state[:, self.target_indices, :]
        source_node_states = state[:, self.source_indices, :]

        # Each row is a pair [target_node, source_node].
        batch = torch.cat((target_node_states, source_node_states), dim=1)

        # Pass messages from source nodes to target nodes.
        messages = self.msg(batch.view(-1, self.obj_dim * 2)).view(
            batch_size, -1, self.obj_dim
        )

        # Weight messages by the probability mass on the edge
        weighted_messages = torch.mul(probs.view(*probs.shape, 1), messages).view(
            batch_size, -1, self.num_obj - 1, self.obj_dim
        )

        # Aggregate the messages
        aggregated = torch.sum(weighted_messages, dim=2)

        # Apply the final function.
        state_and_msg = torch.cat((state, aggregated), dim=2)

        output = self.final(state_and_msg.view(-1, self.obj_dim * 2))
        return output.view(batch_size, -1, self.obj_dim)
