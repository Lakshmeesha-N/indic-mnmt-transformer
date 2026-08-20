import torch
import torch.nn as nn

from models.config import MNMTConfig


class FeedForward(nn.Module):
    """Standard position-wise feed-forward network (baseline)."""

    def __init__(self, config: MNMTConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return ff_output