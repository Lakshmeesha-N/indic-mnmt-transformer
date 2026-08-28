"""
models/config.py

Central configuration for the from-scratch Transformer MNMT model.
Add new fields here as we build encoder, decoder, attention, etc.
Import this config wherever hyperparameters are needed, instead of
hardcoding values inside individual model files.
"""

from dataclasses import dataclass


@dataclass
class MNMTConfig:
    # === Vocabulary (matching IndicTrans2's SRC/TGT tokenizers) ===
    src_vocab_size: int = 32322
    """Vocabulary size of the English (source) tokenizer (dict.SRC.json)."""

    tgt_vocab_size: int = 122672
    """Vocabulary size of the Hindi/Kannada/Tamil (target) tokenizer (dict.TGT.json)."""

    pad_token_id: int = 1
    """Token ID used for padding. <pad> = id 1 in both dict.SRC.json and dict.TGT.json.
    Passed to nn.Embedding(padding_idx=...) and used in attention masks and loss ignore_index."""

    # === Core Model Dimensions ===
    d_model: int = 512
    """Dimensionality of token embeddings and hidden states."""

    max_seq_len: int = 512
    """Maximum sequence length supported (matches MAX_TOKENIZE_LEN in preprocess.py)."""

    # === Attention & Rotary Position Embedding (RoPE) ===
    num_attention_heads: int = 8
    """Number of attention heads in multi-head attention."""

    rope_theta: float = 10000.0
    """Base frequency constant for rotary position embeddings."""

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head (d_model // num_attention_heads)."""
        assert self.d_model % self.num_attention_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        )
        return self.d_model // self.num_attention_heads

    # === Transformer Layers & Feed-Forward ===
    num_encoder_layers: int = 6
    """Number of stacked encoder blocks."""

    num_decoder_layers: int = 6
    """Number of stacked decoder blocks."""

    d_ff: int = 2048
    """Hidden dimension of the position-wise feed-forward layer."""

    # === Split/MoE-style Feed-Forward (novel technique experiment) ===
    use_split_ffn: bool = False
    """If True, use SplitFeedForward (multi-expert FFN) instead of standard FeedForward."""

    num_splits: int = 4
    """Number of parallel expert branches to split d_model into (must evenly divide d_model)."""

    # === Regularization & Loss ===
    dropout: float = 0.1
    """Dropout probability used throughout the model."""

    label_smoothing: float = 0.1
    """Label smoothing factor for the training loss (helps BLEU)."""


# default config instance, importable directly
mnmt_config = MNMTConfig()
