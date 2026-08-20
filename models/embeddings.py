import math
import torch.nn as nn

from models.config import MNMTConfig


class InputEmbed(nn.Module):
    """
    Converts token IDs into scaled dense embedding vectors.
    Used separately for source (English) and target (Hindi/Kannada/Tamil) sides,
    each with its own vocab_size, as configured in MNMTConfig.
    """

    def __init__(self, config: MNMTConfig, vocab_size: int):
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

    def forward(self, x):
        embedded_x = self.embedding(x) * math.sqrt(self.d_model)
        return embedded_x