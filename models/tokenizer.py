"""
models/tokenizer.py

Wraps IndicTrans2's native SentencePiece models + vocab dicts (SRC/TGT) into
one reusable class. Used by preprocess.py (encoding raw text -> token IDs)
and by train.py / inference.py (decoding predicted token IDs -> text, for
BLEU/chrF++ evaluation and real translation output).
"""

import json
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor

MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"

# IndicTrans2 standard language codes:
# Source is English (eng_Latn), Targets: Hindi (hin_Deva), Kannada (kan_Knda), Tamil (tam_Taml)
LANG_CODES = {
    "hi": ("eng_Latn", "hin_Deva"),
    "kn": ("eng_Latn", "kan_Knda"),
    "ta": ("eng_Latn", "tam_Taml"),
}

MAX_TOKENIZE_LEN = 512


class IndicTransTokenizerEngine:
    def __init__(self, model_name: str = MODEL_NAME, token: str = None):
        print(f"Loading IndicTrans2 vocab and SentencePiece models from {model_name} ...")
        src_vocab_path = hf_hub_download(model_name, "dict.SRC.json", token=token)
        tgt_vocab_path = hf_hub_download(model_name, "dict.TGT.json", token=token)
        src_spm_path = hf_hub_download(model_name, "model.SRC", token=token)
        tgt_spm_path = hf_hub_download(model_name, "model.TGT", token=token)

        with open(src_vocab_path, "r", encoding="utf-8") as f:
            self.src_vocab = json.load(f)
        with open(tgt_vocab_path, "r", encoding="utf-8") as f:
            self.tgt_vocab = json.load(f)

        self.src_spm = SentencePieceProcessor(model_file=src_spm_path)
        self.tgt_spm = SentencePieceProcessor(model_file=tgt_spm_path)

        self.src_bos_id = self.src_vocab.get("<s>", 0)
        self.src_pad_id = self.src_vocab.get("<pad>", 1)
        self.src_eos_id = self.src_vocab.get("</s>", 2)
        self.src_unk_id = self.src_vocab.get("<unk>", 3)

        self.tgt_bos_id = self.tgt_vocab.get("<s>", 0)
        self.tgt_pad_id = self.tgt_vocab.get("<pad>", 1)
        self.tgt_eos_id = self.tgt_vocab.get("</s>", 2)
        self.tgt_unk_id = self.tgt_vocab.get("<unk>", 3)

        # reverse lookup tables (id -> piece), built once, used by decode methods
        self._src_id_to_piece = {v: k for k, v in self.src_vocab.items()}
        self._tgt_id_to_piece = {v: k for k, v in self.tgt_vocab.items()}

        print(f"Loaded src_vocab ({len(self.src_vocab)} tokens), tgt_vocab ({len(self.tgt_vocab)} tokens).")
        print("Language tag IDs in src_vocab:")
        for lang, (src_tag, tgt_tag) in LANG_CODES.items():
            print(f"  {lang}: {src_tag} -> {self.src_vocab.get(src_tag)}, {tgt_tag} -> {self.tgt_vocab.get(tgt_tag)}")

    # -----------------------------------------------------------------
    # Encoding (text -> token IDs)
    # -----------------------------------------------------------------
    def encode_src_batch(self, src_lang: str, tgt_lang: str, texts: list, max_len: int = MAX_TOKENIZE_LEN):
        """Tags + tokenizes a batch of English source sentences.

        Produces sequences of the form:
          [src_lang_tag, tgt_lang_tag, tok₁, tok₂, ..., tokₙ, EOS]

        No BOS is prepended — the language tags serve as the leading signal,
        and the encoder reads the full sequence at once (non-autoregressive),
        so it does not need a BOS the way the decoder does.
        """
        results = []
        for text in texts:
            pieces = [src_lang, tgt_lang] + self.src_spm.encode(str(text), out_type=str)
            ids = [self.src_vocab.get(p, self.src_unk_id) for p in pieces]
            if len(ids) > max_len - 1:      # -1 reserves space for EOS only
                ids = ids[: max_len - 1]
            ids = ids + [self.src_eos_id]   # [lang_tags, tokens, EOS]
            results.append(ids)
        return results

    def encode_tgt_batch(self, texts: list, max_len: int = MAX_TOKENIZE_LEN):
        """Tokenizes a batch of Hindi/Kannada/Tamil target sentences.

        Produces sequences of the form: [BOS, tok₁, tok₂, ..., tokₙ, EOS]
        where BOS = <s> (id=0) and EOS = </s> (id=2).

        This ensures teacher-forcing in run_epoch() works correctly:
          decoder_input = tgt_ids[:, :-1]  → [BOS, tok₁, ..., tokₙ]
          labels        = tgt_ids[:, 1:]   → [tok₁, ..., tokₙ, EOS]
        and greedy_decode() can start from the same BOS token.
        """
        bos_id = self.tgt_bos_id   # <s> (id=0) as distinct BOS token
        results = []
        for text in texts:
            pieces = self.tgt_spm.encode(str(text), out_type=str)
            ids = [self.tgt_vocab.get(p, self.tgt_unk_id) for p in pieces]
            if len(ids) > max_len - 2:      # -2 reserves space for BOS + EOS
                ids = ids[: max_len - 2]
            ids = [bos_id] + ids + [self.tgt_eos_id]   # [BOS, tok₁, …, tokₙ, EOS]
            results.append(ids)
        return results

    # -----------------------------------------------------------------
    # Decoding (token IDs -> text) -- used for BLEU/chrF++ eval and inference
    # -----------------------------------------------------------------
    def decode_tgt_batch(self, id_sequences: list, stop_at_eos: bool = True) -> list:
        """
        Converts model-predicted (or ground-truth) target token ID sequences
        back into readable text.

        Args:
            id_sequences: list of lists of token IDs (e.g. model.generate() output)
            stop_at_eos: if True, truncate each sequence at the first </s> token
        Returns:
            list of decoded strings, one per input sequence
        """
        texts = []
        for ids in id_sequences:
            if stop_at_eos and self.tgt_eos_id in ids:
                ids = ids[: ids.index(self.tgt_eos_id)]
            pieces = [self._tgt_id_to_piece.get(i, "<unk>") for i in ids]
            text = self.tgt_spm.decode(pieces)
            texts.append(text)
        return texts

    def decode_src_batch(self, id_sequences: list, stop_at_eos: bool = True) -> list:
        """Same as decode_tgt_batch but for the English/source side (rarely
        needed, included for completeness/debugging)."""
        texts = []
        for ids in id_sequences:
            if stop_at_eos and self.src_eos_id in ids:
                ids = ids[: ids.index(self.src_eos_id)]
            pieces = [self._src_id_to_piece.get(i, "<unk>") for i in ids]
            text = self.src_spm.decode(pieces)
            texts.append(text)
        return texts