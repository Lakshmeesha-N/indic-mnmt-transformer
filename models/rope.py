import torch
import torch.nn as nn
from models.config import MNMTConfig


class SimpleRotaryEmbedding(nn.Module):
    """
    Simple Rotary Position Embedding (RoPE) without caching.
    Computes frequencies on-the-fly to save memory (optimized for T4 GPU training).
    
    Supports input tensors:
      - [batch_size, seq_len, num_heads, head_dim]
    """

    def __init__(self, config: MNMTConfig):
        super().__init__()
        self.dim = config.head_dim
        self.theta = config.rope_theta

        # inv_freq: [dim // 2]
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input query or key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            position_ids: Optional position indices of shape [batch_size, seq_len]
        Returns:
            Rotated tensor with identical shape [batch_size, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        device = x.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

        # Compute frequencies on-the-fly for dim // 2
        # position_ids: [batch_size, seq_len] -> t: [batch_size, seq_len, 1]
        t = position_ids.float().unsqueeze(-1)

        # freqs: [batch_size, seq_len, dim // 2]
        freqs = t * self.inv_freq.to(device)

        # Reshape freqs to match [batch_size, seq_len, 1, dim // 2] for broadcasting across num_heads
        freqs = freqs.unsqueeze(2)

        # Compute cos and sin in input dtype
        cos = torch.cos(freqs).to(x.dtype)
        sin = torch.sin(freqs).to(x.dtype)

        # Split x into first half and second half
        x1 = x[..., : self.dim // 2]  # [batch_size, seq_len, num_heads, dim // 2]
        x2 = x[..., self.dim // 2 :]  # [batch_size, seq_len, num_heads, dim // 2]

        # Apply 2D Givens rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.cat((rotated_x1, rotated_x2), dim=-1)
