import torch
import torch.nn as nn


from models.config import MNMTConfig


class SplitFeedForward(nn.Module):
    """
    Multi-branch (expert-style) feed-forward network.

    Splits the d_model dimension into `num_splits` equal chunks, each processed
    by its own small independent feed-forward branch, then concatenated back.

    dff_per_split is derived from config.d_ff (not hardcoded), so total FFN
    capacity stays consistent with the baseline FeedForward's d_ff budget,
    just distributed across parallel branches instead of one large branch.
    """

    def __init__(self, config: MNMTConfig):
        super().__init__()
        assert config.d_model % config.num_splits == 0, \
            "d_model must be evenly divisible by num_splits"

        self.num_splits = config.num_splits
        self.split_dim = config.d_model // config.num_splits
        self.dff_per_split = config.d_ff // config.num_splits  # derived, not hardcoded

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.split_dim, self.dff_per_split),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.dff_per_split, self.split_dim),
            )
            for _ in range(self.num_splits)
        ])

    def forward(self, x):
        chunks = x.chunk(self.num_splits, dim=-1)
        outputs = [expert(chunk) for expert, chunk in zip(self.experts, chunks)]
        return torch.cat(outputs, dim=-1)