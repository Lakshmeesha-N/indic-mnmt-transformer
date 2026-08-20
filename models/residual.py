import torch.nn as nn
from models.norm import RMSNorm  # or LayerNorm, pick one consistently


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x, sublayer):
        sublayer_output = sublayer(self.norm(x))
        residual_output = x + self.dropout(sublayer_output)
        return residual_output