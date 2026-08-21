import torch


def create_padding_mask(token_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Builds a padding mask: True where token is real, False where it's <pad>.

    Args:
        token_ids: [batch, seq_len]
        pad_token_id: the ID used for padding (config.pad_token_id)

    Returns:
        mask: [batch, 1, 1, seq_len] -- broadcastable over attention scores
              [batch, num_heads, q_len, k_len]
    """
    mask = (token_ids != pad_token_id)              # [batch, seq_len]
    return mask.unsqueeze(1).unsqueeze(2)             # [batch, 1, 1, seq_len]


def create_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Builds a causal (lower-triangular) mask so each position can only attend
    to itself and earlier positions -- prevents the decoder from "seeing"
    future tokens during training.

    Returns:
        mask: [1, 1, seq_len, seq_len] -- broadcastable over attention scores
              [batch, num_heads, q_len, k_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def create_decoder_self_attn_mask(token_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Combines causal mask + padding mask for decoder self-attention.
    A position is attendable only if it's both (a) not in the future and
    (b) not a padding token.

    Args:
        token_ids: [batch, seq_len]  (decoder/target token IDs)

    Returns:
        mask: [batch, 1, seq_len, seq_len]
    """
    batch_size, seq_len = token_ids.shape
    device = token_ids.device

    causal = create_causal_mask(seq_len, device=device)           # [1, 1, seq_len, seq_len]
    padding = create_padding_mask(token_ids, pad_token_id)         # [batch, 1, 1, seq_len]

    combined = causal & padding   # broadcasts to [batch, 1, seq_len, seq_len]
    return combined