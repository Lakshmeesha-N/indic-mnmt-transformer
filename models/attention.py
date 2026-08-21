import math
import torch
import torch.nn as nn

from models.config import MNMTConfig
from models.rope import SimpleRotaryEmbedding


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embeddings (RoPE) applied to Q and K.

    Used for:
      - Encoder self-attention (src attends to src) -> pass padding_mask
      - Decoder masked self-attention (tgt attends to tgt) -> pass causal_mask & padding_mask
      - Decoder cross-attention (tgt attends to encoder output) -> pass encoder padding_mask

    Masking is generic: this class does not build masks itself. The caller
    (encoder.py / decoder.py) builds the appropriate mask using masks.py and
    passes it in via the `mask` argument.
    """

    def __init__(self, config: MNMTConfig):
        super().__init__()
        assert config.d_model % config.num_attention_heads == 0, \
            "d_model must be divisible by num_attention_heads"

        self.d_model = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.rope = SimpleRotaryEmbedding(config)

    def _split_heads(self, x, batch_size, seq_len):
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, head_dim]
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _merge_heads(self, x, batch_size, seq_len):
        # [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, d_model]
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query_input, key_input, value_input, mask=None, apply_rope=True):
        """
        Args:
            query_input: [batch, q_len, d_model]
            key_input:   [batch, k_len, d_model]
            value_input: [batch, k_len, d_model]
            mask: broadcastable to [batch, 1, q_len, k_len]. True/1 = keep, False/0 = mask out.
            apply_rope: apply RoPE to Q/K. True for self-attention, typically False
                        for cross-attention (Q/K come from different sequences).
        Returns:
            output: [batch, q_len, d_model]
        """
        batch_size, q_len, _ = query_input.shape
        k_len = key_input.shape[1]

        q = self.q_proj(query_input)
        k = self.k_proj(key_input)
        v = self.v_proj(value_input)

        q = self._split_heads(q, batch_size, q_len)
        k = self._split_heads(k, batch_size, k_len)
        v = self._split_heads(v, batch_size, k_len)

        if apply_rope:
            q = self.rope(q)
            k = self.rope(k)

        q = q.transpose(1, 2)  # [batch, num_heads, q_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch, num_heads, q_len, k_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [batch, num_heads, q_len, head_dim]
        context = context.transpose(1, 2)          # [batch, q_len, num_heads, head_dim]
        context = self._merge_heads(context, batch_size, q_len)

        output = self.out_proj(context)
        return output