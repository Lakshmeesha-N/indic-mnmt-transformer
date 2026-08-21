"""
scripts/data_utils.py

Loads tokenized .src.ids / .tgt.ids files (produced by preprocess.py) for the
3 known language pairs (hi, kn, ta), combines them into one multilingual
dataset, and provides a Collator class for padding batches.

Per-language validation splits are available via MNMTDataset.per_language_splits()
which returns a dict of {lang_code: MNMTDataset} — each dataset contains only
that language's val pairs (shuffled, then last val_fraction sliced off).
"""

import os
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

LANGS = ["hi", "kn", "ta"]


def _load_ids(path):
    sequences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ids = [int(tok) for tok in line.strip().split()]
            if ids:
                sequences.append(torch.tensor(ids, dtype=torch.long))
    return sequences


class MNMTDataset(Dataset):
    """
    Loads samanantar_hi/kn/ta .src.ids / .tgt.ids from tok_train_dir and
    combines them into one multilingual dataset.

    Two construction modes:
      1. MNMTDataset(tok_train_dir, val_fraction=0.02)  — combined training set (excluding val split)
      2. MNMTDataset.per_language_splits(tok_dir, vf)   — dict of per-language val datasets
    """

    def __init__(self, tok_train_dir, val_fraction: float = 0.02, is_train: bool = True, _pairs=None, seed: int = 42):
        """
        Args:
            tok_train_dir: path to directory containing .src.ids / .tgt.ids files.
            val_fraction: fraction of each language's data held out for validation.
            is_train: if True, keeps only the first (1 - val_fraction) pairs for training.
            _pairs: optional pre-built list of (src_tensor, tgt_tensor) tuples.
            seed: random seed used for deterministic train/val splitting.
        """
        if _pairs is not None:
            # Internal path: constructed from pre-sliced list of pairs
            self.src_data = [p[0] for p in _pairs]
            self.tgt_data = [p[1] for p in _pairs]
            return

        self.src_data = []
        self.tgt_data = []
        rng = random.Random(seed)

        for code in LANGS:
            src_path = os.path.join(tok_train_dir, f"samanantar_{code}.src.ids")
            tgt_path = os.path.join(tok_train_dir, f"samanantar_{code}.tgt.ids")

            src = _load_ids(src_path)
            tgt = _load_ids(tgt_path)
            assert len(src) == len(tgt), f"Mismatched lines: {src_path} vs {tgt_path}"

            pairs = list(zip(src, tgt))
            rng.shuffle(pairs)

            n_val = max(1, int(len(pairs) * val_fraction)) if val_fraction > 0 else 0
            if is_train and n_val > 0:
                selected_pairs = pairs[:-n_val]
                print(f"  loaded {code} (train): {len(selected_pairs)} pairs (held out {n_val} for val)")
            else:
                selected_pairs = pairs
                print(f"  loaded {code}: {len(selected_pairs)} pairs")

            self.src_data.extend([p[0] for p in selected_pairs])
            self.tgt_data.extend([p[1] for p in selected_pairs])

        print(f"  total combined pairs: {len(self.src_data)}")

    @classmethod
    def per_language_splits(cls, tok_dir, val_fraction=0.02, langs=LANGS, seed=42):
        """
        Builds one small validation dataset per language by slicing the last
        val_fraction of each language's data (after shuffling for representativeness).

        Args:
            tok_dir:      directory containing samanantar_{lang}.src.ids / .tgt.ids
            val_fraction: fraction of each language's data to use as val set
            langs:        list of language codes to build splits for
            seed:         RNG seed — fixed so the same pairs are held out every run

        Returns:
            dict mapping lang_code -> MNMTDataset (one dataset per language)
        """
        splits = {}
        rng = random.Random(seed)

        for code in langs:
            src_path = os.path.join(tok_dir, f"samanantar_{code}.src.ids")
            tgt_path = os.path.join(tok_dir, f"samanantar_{code}.tgt.ids")

            src = _load_ids(src_path)
            tgt = _load_ids(tgt_path)
            assert len(src) == len(tgt), f"Mismatched lines: {src_path} vs {tgt_path}"

            pairs = list(zip(src, tgt))
            rng.shuffle(pairs)
            n_val = max(1, int(len(pairs) * val_fraction))
            val_pairs = pairs[-n_val:]

            splits[code] = cls(tok_train_dir=None, _pairs=val_pairs)
            print(f"  val split [{code}]: {len(val_pairs)} pairs")

        return splits

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


class Collator:
    """
    Callable collate class that pads variable-length sequences within a batch.
    Stores pad_token_id once at construction, so it can be passed directly to
    DataLoader(collate_fn=...) without needing functools.partial.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_token_id)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_token_id)
        return src_padded, tgt_padded