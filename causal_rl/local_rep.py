"""Local representations."""

import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple, List


class LocalRep(nn.Module):
    """A local representation.

    Attributes:
        graph_predictor (nn.Module):
        state_predictor (nn.Module):
    """

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GraphAndMessages(nn.Module):
    """A neural network that predicts existence and weight of edges."""

    def __init__(self, num_obj: int, obj_dim: int, layer_widths: Tuple[int, ...]=(32, 32)):
        super().__init__()
        self.num_obj = num_obj
        self.obj_dim = obj_dim

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

        # Final sigmoid to get probabilities.
        layers.append(nn.Sigmoid())

        self.module: nn.Module = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reshaped = inputs.view((-1, self.num_objects, self.obj_dim))
        # Pair up source and target states
        paired_states = torch.cat((reshaped[:, self.source_indices], reshaped[:, self.sink_indices]), dim=2)
        output = self.module(paired_states).squeeze()

        edges = output[0: len(output) - 1: 2]
        messages = output[1: len(output): 2]

        return edges, messages