"""
models/transformer.py

Full Encoder-Decoder Transformer architecture for Multilingual Neural Machine Translation (MNMT).
Combines:
  - Input embeddings (source and target vocabularies)
  - Encoder stack with RMSNorm and RoPE multi-head attention
  - Decoder stack with causal self-attention and encoder cross-attention
  - Output projection (Linear generator) to target vocabulary
"""

import torch.nn as nn

from models.config import MNMTConfig
from models.embeddings import InputEmbed
from models.encoder import Encoder
from models.decoder import Decoder
from models.mask import create_padding_mask, create_decoder_self_attn_mask


class Transformer(nn.Module):
    """
    Multilingual Transformer model for translation from English into Indic languages.
    """

    def __init__(self, config: MNMTConfig):
        super().__init__()
        self.config = config

        self.src_embed = InputEmbed(config, config.src_vocab_size)
        self.tgt_embed = InputEmbed(config, config.tgt_vocab_size)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.generator = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)

    def encode(self, src_ids):
        """Encodes source token IDs into contextual hidden states."""
        src_mask = create_padding_mask(src_ids, self.config.pad_token_id)
        src_emb = self.src_embed(src_ids)
        return self.encoder(src_emb, src_mask=src_mask)

    def decode(self, tgt_ids, encoder_output, src_ids):
        """Decodes target token IDs given encoder representations."""
        tgt_mask = create_decoder_self_attn_mask(tgt_ids, self.config.pad_token_id)
        cross_mask = create_padding_mask(src_ids, self.config.pad_token_id)
        tgt_emb = self.tgt_embed(tgt_ids)
        return self.decoder(tgt_emb, encoder_output, tgt_mask=tgt_mask, cross_mask=cross_mask)

    def forward(self, src_ids, tgt_ids):
        """
        Forward pass for training and evaluation.

        Args:
            src_ids: [batch_size, src_len]
            tgt_ids: [batch_size, tgt_len] (decoder input)

        Returns:
            logits: [batch_size, tgt_len, tgt_vocab_size]
        """
        src_mask = create_padding_mask(src_ids, self.config.pad_token_id)
        tgt_mask = create_decoder_self_attn_mask(tgt_ids, self.config.pad_token_id)
        cross_mask = create_padding_mask(src_ids, self.config.pad_token_id)

        src_emb = self.src_embed(src_ids)
        encoder_output = self.encoder(src_emb, src_mask=src_mask)

        tgt_emb = self.tgt_embed(tgt_ids)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask=tgt_mask, cross_mask=cross_mask)

        return self.generator(decoder_output)
