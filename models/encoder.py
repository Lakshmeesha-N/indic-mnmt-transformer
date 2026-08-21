import torch.nn as nn

from models.config import MNMTConfig
from models.attention import MultiHeadAttention
from models.feedforward import FeedForward
from models.moefeedforward import SplitFeedForward
from models.residual import ResidualConnection
from models.norm import RMSNorm


class EncoderLayer(nn.Module):
    """
    One encoder block:
      x -> ResidualConnection(self_attention) -> ResidualConnection(feed_forward) -> output

    Uses pre-norm residual connections (norm applied inside ResidualConnection
    before the sublayer, not after).
    """

    def __init__(self, config: MNMTConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = SplitFeedForward(config) if config.use_split_ffn else FeedForward(config)

        self.residual1 = ResidualConnection(config.d_model, config.dropout)
        self.residual2 = ResidualConnection(config.d_model, config.dropout)

    def forward(self, x, src_mask=None):
        # self-attention sublayer (encoder attends to itself, RoPE applied)
        x = self.residual1(x, lambda x_norm: self.self_attention(
            x_norm, x_norm, x_norm, mask=src_mask, apply_rope=True
        ))

        # feed-forward sublayer
        x = self.residual2(x, self.feed_forward)

        return x


class Encoder(nn.Module):
    """
    Stack of N EncoderLayers, followed by a final normalization layer.
    """

    def __init__(self, config: MNMTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.final_norm(x)