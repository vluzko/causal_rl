"""Algorithms for predicting the causal diagram governing an environment at a particular time step.

"""
import numpy as np
import torch

from einops import rearrange
from gym import Space
from torch import nn, optim
from torch.nn import functional as F
from typing import Tuple, List, Optional


from causal_rl import utils


class CDAGPredictor(nn.Module):
    """Abstract class for predicting a causal DAG from state.

    Attributes:
        num_obj (int):                                  The number of objects in the state.
        obj_dim (int):                                  The dimension of an object.
        asymmetric (bool):                              Whether or not A affects B implies B affects A.
    """

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        asymmetric: bool = False,
    ):
        super().__init__()
        self.num_obj = num_obj
        self.obj_dim = obj_dim
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

    def __init__(
        self,
        num_obj: int,
        unconstrained_layers: nn.Module,
        location_indices: Tuple[int, ...],
        max_distance: float,
    ):
        super().__init__()
        self.num_obj = num_obj
        self.unconstrained_layers = unconstrained_layers
        self.location_indices = location_indices
        self.max_distance = max_distance
        (
            self.source_state_indices,
            self.target_state_indices,
        ) = utils.state_to_source_sink_indices(self.num_obj)

    def distances(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        positions = state.view(batch_size, self.num_obj, -1)[
            :, :, self.location_indices
        ]

        source_node_states = positions[:, self.source_state_indices].view(
            -1, len(self.location_indices)
        )
        target_node_states = positions[:, self.target_state_indices].view(
            -1, len(self.location_indices)
        )

        distances = F.pairwise_distance(source_node_states, target_node_states)
        return distances.view(batch_size, -1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        unconstrained = self.unconstrained_layers(inputs)
        distances = self.distances(inputs)
        in_range = distances < self.max_distance
        return in_range * unconstrained.squeeze()


class Localizable(CDAGPredictor):
    """A CDAG predictor that can have a local connections only inductive bias.

    Attributes:
        location_indices: (Optional[Tuple[int, ...]]):  The indices of state that correspond to location.
        max_distance (Optional[float]):                 The maximum distance between nodes with a causal connection.
    """

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        asymmetric: bool = False,
        max_distance: Optional[float] = None,
        location_indices: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(num_obj, obj_dim, asymmetric=asymmetric)
        self.max_distance = max_distance
        self.location_indices = location_indices


class FullyConnectedPredictor(Localizable):
    """A fully connected network for predicting connections between nodes.
    Takes the entire state as input, and outputs a probability for each pair of distinct nodes.

    The same output is given to the edge [a, b] and to [b, a]

    Attributes:
        layers (nn.Module): The underlying neural network + any additional transformations required.
    """

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        layer_widths: Tuple[int, ...] = (32, 32),
        asymmetric: bool = False,
        max_distance: Optional[float] = None,
        location_indices: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(
            num_obj,
            obj_dim,
            asymmetric=asymmetric,
            location_indices=location_indices,
            max_distance=max_distance,
        )
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
            assert (
                self.location_indices is not None
            ), "For a local predictor you must pass location indices."
            self.layers = LocalOnly(
                self.num_obj, unconstrained, self.location_indices, self.max_distance
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layers(inputs)
        return output


class ConvLinkPredictor(Localizable):
    """A link predictor for predicting edges between nodes.
    Takes the entire state as input, splits it into pairs of distinct nodes, then runs the same
    network on all pairs to produce a probability of connection.
    """

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        layer_widths: Tuple[int, ...] = (32, 32),
        asymmetric: bool = False,
        max_distance: Optional[float] = None,
        location_indices: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(
            num_obj,
            obj_dim,
            asymmetric=asymmetric,
            location_indices=location_indices,
            max_distance=max_distance,
        )
        self.input_size = 2 * obj_dim
        self.output_size = 1
        self.layer_widths = layer_widths
        self.number_of_pairs = (
            self.num_obj * (self.num_obj - 1)
            if self.asymmetric
            else (self.num_obj * (self.num_obj - 1)) // 2
        )

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
            self.source_indices, self.sink_indices = utils.state_to_source_sink_indices(
                self.num_obj
            )
        else:
            # If symmetric, we look at all unordered pairs of states.
            (
                self.source_indices,
                self.sink_indices,
            ) = utils.flattened_upper_adj_to_adj_indices(self.num_obj)
            layers.append(SymToFull(self.num_obj))

        if self.max_distance is None:
            self.module: nn.Module = nn.Sequential(*layers)
        else:
            unconstrained = nn.Sequential(*layers)
            assert (
                self.location_indices is not None
            ), "For a local predictor you must pass location indices."
            self.module = LocalOnly(
                self.num_obj, unconstrained, self.location_indices, self.max_distance
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reshaped = inputs.view((-1, self.num_obj, self.obj_dim))
        # Pair up source and target states
        paired_states = torch.cat(
            (reshaped[:, self.source_indices], reshaped[:, self.sink_indices]), dim=2
        )
        output = self.module(paired_states)
        # The module produces a tensor of shape (batch_size x (num_obj * (num_obj - 1)) x 1)
        return output.squeeze()


def init(module, weight_init, bias_init, gain=1):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class EmptyGraphPredictor(nn.Module):
    """A graph predictor that always predicts an empty graph.
    Used as a baseline.
    """

    def __init__(self, input_dim: int, graph_dim: int):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 0


class AttentionGraphPredictor(CDAGPredictor):
    """Attention-based graph predictor.

    Attributes:
        num_heads:      The number of heads for multi-headed attention.
        dim_head:       The dimension of each head.
        dim_linear:     The dimension of the linear layer in the transformer block.
        dropout:        Dropout fraction.
        num_layers:     The number of transformer block layers.
        sigmoid:
    """

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        num_heads: int = 4,
        dim_head: int = 64,
        dim_linear: int = 1024,
        dropout: float = 0.1,
        num_layers: int = 4,
        sigmoid: bool = True,
    ):
        super().__init__(num_obj, obj_dim, asymmetric=True)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_linear = dim_linear
        self.dropout = dropout
        self.num_layers = num_layers
        self.transformer_stack = nn.ModuleList(
            [
                TransformerBlock(obj_dim, num_heads, dim_head, dim_linear, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder = InnerProductDecoder(sigmoid=sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer_stack:
            x = layer(x)
        x = self.decoder(x)
        return x


class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    """

    def __init__(self, dim, heads=8, dim_head=64, dim_linear_block=1024, dropout=0.1):
        """
        Args:
            dim: token vector length
            heads: number of heads
            dim_head: dimension of head
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
        """
        super().__init__()
        self.mhsa = MultiHeadAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = self.mhsa(inputs, mask)
        y = self.norm_1(self.drop(output) + inputs)
        return self.norm_2(self.linear(y) + y)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        # inner_dim is the dimension of the tokens
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5  # 1/sqrt(dim)
        self.to_qkv = nn.Linear(
            dim, inner_dim * 3, bias=False
        )  # Wq, Wk, Wv for each vector, hence *3
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        qkv = self.to_qkv(inputs)

        # split into multi head attentions
        q, k, v = rearrange(
            qkv, "b n (h qkv d) -> b h n qkv d", h=self.heads, qkv=3
        ).unbind(dim=-2)

        # Batch matrix multiplication by QK^t and scaling
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if mask is not None:
            mask_value = torch.finfo(dots.dtype).min
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            dots.masked_fill_(~mask, mask_value)

        # follow the softmax,q,d,v equation in the paper
        # softmax along row axis of the attention card
        attn = dots.softmax(dim=-1)

        # product of v times whatever inside softmax
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        # concat heads into one matrix
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MLPBase(nn.Module):
    """Basic multi-layer linear model."""

    def __init__(self, num_inputs: int, num_outputs: int, hidden_size: int = 64):
        super().__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        init2_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.model = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init2_(nn.Linear(hidden_size, num_outputs)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class DenseNetNoDiag(CDAGPredictor):
    """A dense net graph predictor that does not force the diagonal to be non-zero"""

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        hidden_dim: int,
        num_layers: int,
        sigmoid: bool = True,
    ):
        super().__init__(num_obj, obj_dim, asymmetric=True)
        self.input_dim = self.num_obj * self.obj_dim
        self.hidden_dim = hidden_dim
        self.dense_net = D2RLNet(
            self.input_dim, self.num_obj**2, hidden_dim, num_layers
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dense_net(inputs)
        x = rearrange(x, "b (g h) -> b g h", g=self.num_obj, h=self.num_obj)
        return torch.sigmoid(x)


class DenseNetGraphPredictor(CDAGPredictor):
    """
    Turns a flat vector of size input_dim into a dense probabilistic graph adjacency matrix of size graph_dim x graph_dim
    """

    def __init__(
        self,
        num_obj: int,
        obj_dim: int,
        hidden_dim: int,
        num_layers: int,
        sigmoid: bool = True,
    ):
        super().__init__(num_obj, obj_dim, asymmetric=True)
        self.input_dim = self.num_obj * self.obj_dim
        self.hidden_dim = hidden_dim
        self.dense_net = D2RLNet(
            self.input_dim, int(self.num_obj * hidden_dim), hidden_dim, num_layers
        )
        self.decoder = InnerProductDecoder(sigmoid=sigmoid)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dense_net(inputs)
        x = rearrange(x, "b (g h) -> b g h", g=self.num_obj, h=self.hidden_dim)
        x = self.decoder(x)
        return x


class InnerProductDecoder(nn.Module):
    """Graph decoder for using inner product for prediction.
    Requires z to be of shape (batch_size, graph_size, latent_dim)
    """

    def __init__(self, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        prod = torch.einsum("bij,bkj->bik", inputs, inputs)
        return torch.sigmoid(prod) if self.sigmoid else prod


class D2RLNet(nn.Module):
    """Deep dense feedforward network https://arxiv.org/abs/2010.09163"""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.num_layers = num_layers
        in_dim = self.num_inputs + hidden_dim
        self.block_list = [nn.Linear(in_dim, hidden_dim) for _ in range(num_layers)]

        self.apply(weights_init_)

        # Model architecture
        self.input = nn.Linear(self.num_inputs, hidden_dim)
        self.body = nn.ModuleList(self.block_list)
        self.output = nn.Linear(hidden_dim, self.num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        xu = inputs
        xu = torch.flatten(xu, start_dim=1)
        x1 = F.relu(self.input(xu))
        x1 = torch.cat([x1, xu], dim=1)
        for i, layer in enumerate(self.body):
            if i != len(self.body) - 1:
                x1 = F.relu(layer(x1))
                x1 = torch.cat([x1, xu], dim=1)
            else:
                # last layer, no cat
                x1 = F.relu(layer(x1))
        outputs = self.output(x1)

        return outputs
