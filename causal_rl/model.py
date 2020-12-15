"""A full state model."""
import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, functional as F

from typing import Tuple, List

from causal_rl import utils
from causal_rl.graph_predictor import CDAGPredictor, FullyConnectedPredictor
from causal_rl.state_predictor import DiscretePredictor, WeightedPredictor


class CausalPredictor(nn.Module):
    """Predict the next state, with an internal causal diagram predictor.

    Attributes:
        num_obj (int):                      The number of objects being modeled
        obj_dim (int):                      The dimension of each object
        graph_predictor (CDAGPredictor):    The causal DAG predictor to use
        state_predictor (nn.Module):        The state predictor to use
    """

    def __init__(self, num_obj: int, obj_dim: int, variational: bool=False):
        self.num_obj = num_obj
        self.obj_dim = obj_dim


    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call the graph predictor, then the state predictor."""
        graph = self.graph_predictor(inputs)


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