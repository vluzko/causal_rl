"""A full state model."""
import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, functional as F

from typing import Tuple, List

from causal_rl import utils
# from causal_rl.graph_predictor import CDAGPredictor, FullyConnectedPredictor
# from causal_rl.state_predictor import DiscretePredictor, WeightedPredictor


class LocalRepresentation(nn.Module):
    """Predict the next state, with an internal causal diagram predictor.

    Attributes:
        num_obj (int):                      The number of objects being modeled.
        obj_dim (int):                      The dimension of each object.
        graph_predictor (nn.Module):        The causal DAG predictor to use.
        state_predictor (nn.Module):        The state predictor to use.
    """

    def __init__(self, num_obj: int, obj_dim: int):
        super().__init__()
        self.num_obj = num_obj
        self.obj_dim = obj_dim

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call the graph predictor, then the state predictor."""
        graph = self.graph_predictor(inputs)  # type: ignore
        raise NotImplementedError

    def losses(self, loss: torch.Tensor, predicted_graph: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute all losses."""
        raise NotImplementedError


class NaivePredictor(nn.Module):
    """Predict the next state from the previous state, using a fully connected network."""
    def __init__(self, num_obj: int, obj_dim: int, layer_widths: Tuple[int, ...]=(32, 32)):
        super().__init__()
        self.num_obj = num_obj
        self.obj_dim = obj_dim
        self.input_size = self.num_obj * self.obj_dim
        self.output_size = self.input_size
        self.layer_widths = (self.input_size, *layer_widths)
        layers: List[nn.Module] = []

        for in_size, out_size in zip(self.layer_widths[:-1], self.layer_widths[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(self.layer_widths[-1], self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        output = self.layers(inputs.view(batch_size, -1))

        return output.view(batch_size, self.num_obj, self.obj_dim)


class GraphAndMessages(LocalRepresentation):
    """A neural network that predicts existence and weight of edges."""

    def __init__(self, num_obj: int, obj_dim: int, layer_widths: Tuple[int, ...]=(32, 32), final_widths: Tuple[int, ...]=(16,)):
        super().__init__(num_obj, obj_dim)

        self.input_size = 2 * obj_dim
        self.output_size = 2
        self.layer_widths = layer_widths
        layers: List[nn.Module] = [nn.Linear(self.input_size, layer_widths[0])]

        for in_size, out_size in zip(layer_widths[:-1], layer_widths[1:]):
            layers.append(nn.BatchNorm1d(in_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(layer_widths[-1], self.output_size))

        self.module: nn.Module = nn.Sequential(*layers)

        final_layers: List[nn.Module] = [nn.Linear(self.obj_dim * 2, final_widths[0])]

        for prev_width, width in zip(final_widths[:-1], final_widths[1:]):
            final_layers.append(nn.BatchNorm1d(prev_width))
            final_layers.append(nn.LeakyReLU())  # type: ignore
            final_layers.append(nn.Linear(prev_width, width))
        final_layers.append(nn.BatchNorm1d(final_widths[-1]))
        final_layers.append(nn.LeakyReLU())
        final_layers.append(nn.Linear(final_widths[-1], self.obj_dim))

        self.final = nn.Sequential(*final_layers)
        self.source_indices, self.target_indices = utils.state_to_source_sink_indices(self.num_obj)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        reshaped = inputs.view((-1, self.num_obj, self.obj_dim))
        # Pair up source and target states
        paired_states = torch.cat((reshaped[:, self.source_indices], reshaped[:, self.target_indices]), dim=2)
        output = self.module(paired_states).squeeze()

        edges = nn.functional.sigmoid(output[0: len(output) - 1: 2])
        messages = output[1: len(output): 2]

        weighted_messages = torch.mul(edges, messages)

        # sources = inputs[:, self.source_indices]
        # targets = inputs[:, self.target_indices]

        # sources_with_edges = torch.mul(sources, edges.view(*edges.shape, 1))
        # targets_with_edges = torch.mul(targets, edges.view(*edges.shape, 1))

        # Actual message passing
        # need_message = torch.cat((sources_with_edges, targets_with_edges), dim=2)
        # messages = self.msg(need_message.view(-1, self.obj_dim * 2))

        reshaped_for_agg = weighted_messages.view(batch_size, inputs.shape[1], -1, inputs.shape[2])
        # Add the messages from each source node to a particular target node.
        # There is a separate channel for each component of the node vector
        aggregated = torch.sum(reshaped_for_agg, dim=2)

        # Final step
        state_and_messages = torch.cat((inputs, aggregated), dim=2)
        output = self.final(state_and_messages.view(-1, self.obj_dim * 2))

        return edges, output.view(batch_size, -1, self.obj_dim)
