"""Algorithms for predicting the causal diagram governing an environment at a particular time step.

"""
import torch

from gym import Space
from torch import nn, optim
from torch.nn import functional as F
from typing import Tuple, List, Optional

from causal_rl import utils
# from causal_rl.environments import SupervisedCDAG


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CDAGPredictor(nn.Module):
    """Abstract class for predicting a causal DAG from state.

    Attributes:
        num_obj (int):                              The number of objects in the state.
        obj_dim (int):                                  The dimension of an object.
        location_indices: (Optional[Tuple[int, ...]]):  The indices of state that correspond to location.
        max_distance (Optional[float]):                 The maximum distance between nodes with a causal connection.
        asymmetric (bool):                              Whether or not A affects B implies B affects A.
    """

    def __init__(self, num_obj: int, obj_dim: int,
                 asymmetric: bool=False,
                 max_distance: Optional[float]=None,
                 location_indices: Optional[Tuple[int, ...]]=None):
        super().__init__()
        self.num_obj = num_obj
        self.obj_dim = obj_dim
        self.location_indices = location_indices
        self.max_distance = max_distance
        self.asymmetric = asymmetric

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: The state at a particular time step.

        Returns:
            A tensor representing the edges between nodes. The order of edges is:
                [[0, 1], ..., [0, n], [1, 0], [1, 2],..., [1, n], ..., [n, 0], ..., [n, n-1]]
        """
        raise NotImplementedError


class SymToFull(nn.Module):
    """A layer to convert an upper diagonal representation of a symmetric matrix to a full matrix."""

    def __init__(self, num_obj: int):
        super().__init__()
        self.num_obj = num_obj
        self.sym_to_asym = utils.symmetric_to_asymmetric_indices(self.num_obj)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, self.sym_to_asym]


class LocalOnly(nn.Module):
    """A layer to restrict predictions to be local."""

    def __init__(self, num_obj: int, unconstrained_layers: nn.Module, location_indices: Tuple[int, ...], max_distance: float):
        super().__init__()
        self.num_obj = num_obj
        self.unconstrained_layers = unconstrained_layers
        self.location_indices = location_indices
        self.max_distance = max_distance
        self.source_state_indices, self.target_state_indices = utils.state_to_source_sink_indices(self.num_obj)

    def distances(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        positions = state.view(batch_size, self.num_obj, -1)[:, :, self.location_indices]

        source_node_states = positions[:, self.source_state_indices].view(-1, len(self.location_indices))
        target_node_states = positions[:, self.target_state_indices].view(-1, len(self.location_indices))

        distances = F.pairwise_distance(source_node_states, target_node_states)
        return distances.view(batch_size, -1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        unconstrained = self.unconstrained_layers(inputs)
        distances = self.distances(inputs)
        in_range = (distances < self.max_distance)
        return in_range * unconstrained.squeeze()


class FullyConnectedPredictor(CDAGPredictor):
    """A fully connected network for predicting connections between nodes.
    Takes the entire state as input, and outputs a probability for each pair of distinct nodes.

    The same output is given to the edge [a, b] and to [b, a]

    Attributes:
        layers (nn.Module): The underlying neural network + any additional transformations required.
    """

    def __init__(self, num_obj: int, obj_dim: int, layer_widths: Tuple[int, ...]=(32, 32),
                 asymmetric: bool=False,
                 max_distance: Optional[float]=None,
                 location_indices: Optional[Tuple[int, ...]]=None):
        super().__init__(num_obj, obj_dim, asymmetric=asymmetric, location_indices=location_indices, max_distance=max_distance)
        self.input_size = num_obj * obj_dim

        if self.asymmetric:
            # If asymmetric, we output one prediction for every ordered pair of distinct objects
            self.output_size = num_obj * (num_obj - 1)
        else:
            # If symmetric, it's one for every unordered pair of distinct objects
            self.output_size = (num_obj * (num_obj - 1)) // 2

        # Set up the core neural network.
        self.layer_widths = layer_widths
        layers: List[nn.Module] = [nn.Linear(self.input_size, layer_widths[0])]

        for prev_width, width in zip(layer_widths[:-1], layer_widths[1:]):
            layers.append(nn.BatchNorm1d(prev_width))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(prev_width, width))

        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(layer_widths[-1], self.output_size))

        # Final sigmoid to get probabilities.
        layers.append(nn.Sigmoid())

        # If it's not asymmetric, we need to expand the upper diagonal representation to a full matrix.
        if not self.asymmetric:
            layers.append(SymToFull(self.num_obj))

        if self.max_distance is None:
            self.layers: nn.Module = nn.Sequential(*layers)
        else:
            unconstrained = nn.Sequential(*layers)
            assert self.location_indices is not None, "For a local predictor you must pass location indices."
            self.layers = LocalOnly(self.num_obj, unconstrained, self.location_indices, self.max_distance)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layers(inputs)
        return output


class ConvLinkPredictor(CDAGPredictor):
    """A link predictor for predicting edges between nodes.
    Takes the entire state as input, splits it into pairs of distinct nodes, then runs the same
    network on all pairs to produce a probability of connection.
    """
    def __init__(self, num_obj: int, obj_dim: int, layer_widths: Tuple[int, ...]=(32, 32),
                 asymmetric: bool=False,
                 max_distance: Optional[float]=None,
                 location_indices: Optional[Tuple[int, ...]]=None):
        super().__init__(num_obj, obj_dim, asymmetric=asymmetric, location_indices=location_indices, max_distance=max_distance)
        self.input_size = 2 * obj_dim
        self.output_size = 1
        self.layer_widths = layer_widths
        self.number_of_pairs = self.num_obj * (self.num_obj - 1) if self.asymmetric else (self.num_obj * (self.num_obj - 1)) // 2

        layers: List[nn.Module] = [nn.Linear(self.input_size, layer_widths[0])]

        for in_size, out_size in zip(layer_widths[:-1], layer_widths[1:]):
            layers.append(nn.BatchNorm1d(self.number_of_pairs))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(layer_widths[-1], self.output_size))

        # Final sigmoid to get probabilities.
        layers.append(nn.Sigmoid())


        if self.asymmetric:
            # If asymmetric, we look at all ordered pairs of states
            self.source_indices, self.sink_indices = utils.state_to_source_sink_indices(self.num_obj)
        else:
            # If symmetric, we look at all unordered pairs of states.
            self.source_indices, self.sink_indices = utils.flattened_upper_adj_to_adj_indices(self.num_obj)
            layers.append(SymToFull(self.num_obj))

        if self.max_distance is None:
            self.module: nn.Module = nn.Sequential(*layers)
        else:
            unconstrained = nn.Sequential(*layers)
            assert self.location_indices is not None, "For a local predictor you must pass location indices."
            self.module = LocalOnly(self.num_obj, unconstrained, self.location_indices, self.max_distance)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reshaped = inputs.view((-1, self.num_obj, self.obj_dim))
        # Pair up source and target states
        paired_states = torch.cat((reshaped[:, self.source_indices], reshaped[:, self.sink_indices]), dim=2)
        output = self.module(paired_states)
        # The module produces a tensor of shape (batch_size x (num_obj * (num_obj - 1)) x 1)
        return output.squeeze()
