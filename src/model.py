import torch
from torch import nn


class GraphNASCell(nn.Module):
    """A single differentiable graph neural architecture cell."""

    def __init__(self, in_dim, out_dim, num_ops=12):
        super().__init__()
        self.ops = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for _ in range(num_ops)]
        )

    def forward(self, x, adjacency):
        messages = [op(x) for op in self.ops]
        return torch.stack(messages, dim=0).mean(dim=0)
