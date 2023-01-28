import torch

from torch import nn, functional as F


class GNN(nn.Module):
    def forward(self, state: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        import pdb

        pdb.set_trace()
