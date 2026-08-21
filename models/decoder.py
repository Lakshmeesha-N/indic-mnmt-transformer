import torch.nn as nn

from models.config import MNMTConfig
from models.attention import MultiHeadAttention
from models.feedforward import FeedForward
from models.moefeedforward import SplitFeedForward
from models.residual import ResidualConnection
from models.norm import RMSNorm


class DecoderLayer(nn.Module):
    """
    One decoder block, with 3 sublayers:
      1. Masked self-attention (tgt attends to tgt, causal + padding mask, RoPE applied)
      2. Cross-attention (tgt attends to encoder output, padding mask only, no RoPE)
      3. Feed-forward

    Uses pre-norm residual connections, same pattern as EncoderLayer.
    """

    def __init__(self, config: MNMTConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward = SplitFeedForward(config) if config.use_split_ffn else FeedForward(config)

        self.residual1 = ResidualConnection(config.d_model, config.dropout)
        self.residual2 = ResidualConnection(config.d_model, config.dropout)
        self.residual3 = ResidualConnection(config.d_model, config.dropout)

    def forward(self, x, encoder_output, tgt_mask=None, cross_mask=None):
        """
        Args:
            x: decoder input [batch, tgt_len, d_model]
            encoder_output: [batch, src_len, d_model]
            tgt_mask: causal + padding mask for decoder self-attention
                      (from masks.create_decoder_self_attn_mask)
            cross_mask: padding mask for encoder output
                        (from masks.create_padding_mask on src_ids)
        """
        # 1. masked self-attention (RoPE applied, since Q/K both come from decoder sequence)
        x = self.residual1(x, lambda x_norm: self.self_attention(
            x_norm, x_norm, x_norm, mask=tgt_mask, apply_rope=True
        ))

        # 2. cross-attention (Q from decoder, K/V from encoder output -- different position
        #    spaces, so RoPE is skipped here)
        x = self.residual2(x, lambda x_norm: self.cross_attention(
            x_norm, encoder_output, encoder_output, mask=cross_mask, apply_rope=False
        ))

        # 3. feed-forward
        x = self.residual3(x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    Stack of N DecoderLayers, followed by a final normalization layer.
    """

    def __init__(self, config: MNMTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_decoder_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)

    def forward(self, x, encoder_output, tgt_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask, cross_mask=cross_mask)
        return self.final_norm(x)