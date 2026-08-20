"""
preprocess.py

Steps:
  1. Load raw TSVs from data/raw/train/ (samanantar) and data/raw/test/ (IN22)
  2. Clean: drop empty/duplicate rows, filter by sentence length
  3. Load IndicTrans2's SentencePiece models and vocabularies from HuggingFace
  4. Prepend src & tgt language tags (e.g., eng_Latn hin_Deva) to source sequences
  5. Tokenize src/tgt pairs into token IDs using SentencePiece and vocabulary maps
  6. Save tokenized output into data/tokenized/train/ and data/tokenized/test/
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "train")
RAW_TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "test")
TOK_TRAIN_DIR = os.path.join(BASE_DIR, "data", "tokenized", "train")
TOK_TEST_DIR = os.path.join(BASE_DIR, "data", "tokenized", "test")

os.makedirs(TOK_TRAIN_DIR, exist_ok=True)
os.makedirs(TOK_TEST_DIR, exist_ok=True)

MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"

# IndicTrans2 standard language codes:
# Source is English (eng_Latn), Targets: Hindi (hin_Deva), Kannada (kan_Knda), Tamil (tam_Taml)
LANG_CODES = {
    "hi": ("eng_Latn", "hin_Deva"),
    "kn": ("eng_Latn", "kan_Knda"),
    "ta": ("eng_Latn", "tam_Taml"),
}

MIN_LEN = 1
MAX_LEN = 300
MAX_TOKENIZE_LEN = 512


class IndicTransTokenizerEngine:
    def __init__(self, model_name: str, token: str = None):
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

        self.src_unk_id = self.src_vocab.get("<unk>", 3)
        self.src_eos_id = self.src_vocab.get("</s>", 2)
        self.tgt_unk_id = self.tgt_vocab.get("<unk>", 3)
        self.tgt_eos_id = self.tgt_vocab.get("</s>", 2)

        print(f"Loaded src_vocab ({len(self.src_vocab)} tokens), tgt_vocab ({len(self.tgt_vocab)} tokens).")
        print("Language tag IDs in src_vocab:")
        for lang, (src_tag, tgt_tag) in LANG_CODES.items():
            print(f"  {lang}: {src_tag} -> {self.src_vocab.get(src_tag)}, {tgt_tag} -> {self.src_vocab.get(tgt_tag)}")

    def encode_src_batch(self, src_lang: str, tgt_lang: str, texts: list, max_len: int = MAX_TOKENIZE_LEN):
        results = []
        for text in texts:
            pieces = [src_lang, tgt_lang] + self.src_spm.encode(str(text), out_type=str)
            ids = [self.src_vocab.get(p, self.src_unk_id) for p in pieces]
            if len(ids) > max_len - 1:
                ids = ids[: max_len - 1]
            ids.append(self.src_eos_id)
            results.append(ids)
        return results

    def encode_tgt_batch(self, texts: list, max_len: int = MAX_TOKENIZE_LEN):
        results = []
        for text in texts:
            pieces = self.tgt_spm.encode(str(text), out_type=str)
            ids = [self.tgt_vocab.get(p, self.tgt_unk_id) for p in pieces]
            if len(ids) > max_len - 1:
                ids = ids[: max_len - 1]
            ids.append(self.tgt_eos_id)
            results.append(ids)
        return results


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["src", "tgt"])
    df["src"] = df["src"].astype(str).str.strip()
    df["tgt"] = df["tgt"].astype(str).str.strip()
    df = df[(df["src"] != "") & (df["tgt"] != "")]
    df = df.drop_duplicates(subset=["src", "tgt"])

    src_len = df["src"].str.split().str.len()
    tgt_len = df["tgt"].str.split().str.len()
    mask = (
        (src_len >= MIN_LEN) & (src_len <= MAX_LEN) &
        (tgt_len >= MIN_LEN) & (tgt_len <= MAX_LEN)
    )
    df = df[mask]

    print(f"    cleaned {before} -> {len(df)} rows")
    return df.reset_index(drop=True)


def tokenize_and_save(df: pd.DataFrame, tokenizer: IndicTransTokenizerEngine, src_lang: str, tgt_lang: str, out_prefix: str):
    src_ids = tokenizer.encode_src_batch(src_lang, tgt_lang, df["src"].tolist())
    tgt_ids = tokenizer.encode_tgt_batch(df["tgt"].tolist())

    with open(out_prefix + ".src.ids", "w", encoding="utf-8") as f:
        for ids in src_ids:
            f.write(" ".join(map(str, ids)) + "\n")

    with open(out_prefix + ".tgt.ids", "w", encoding="utf-8") as f:
        for ids in tgt_ids:
            f.write(" ".join(map(str, ids)) + "\n")

    print(f"    saved {out_prefix}.src.ids / .tgt.ids  ({len(df)} rows)")


def process_train(tokenizer: IndicTransTokenizerEngine):
    print("\nProcessing training data (Samanantar) ...")
    for code, (src_tag, tgt_tag) in LANG_CODES.items():
        path = os.path.join(RAW_TRAIN_DIR, f"samanantar_{code}.tsv")
        if not os.path.exists(path):
            print(f"  [SKIPPED] {path} not found")
            continue
        print(f"  reading {path}")
        df = pd.read_csv(path, sep="\t")
        df = clean_df(df)
        tokenize_and_save(df, tokenizer, src_tag, tgt_tag, os.path.join(TOK_TRAIN_DIR, f"samanantar_{code}"))


def process_test(tokenizer: IndicTransTokenizerEngine):
    print("\nProcessing test data (IN22-Gen, IN22-Conv) ...")
    for split in ["IN22-Gen", "IN22-Conv"]:
        for code, (src_tag, tgt_tag) in LANG_CODES.items():
            path = os.path.join(RAW_TEST_DIR, f"{split}_{code}.tsv")
            if not os.path.exists(path):
                print(f"  [SKIPPED] {path} not found")
                continue
            print(f"  reading {path}")
            df = pd.read_csv(path, sep="\t")
            df = clean_df(df)
            tokenize_and_save(df, tokenizer, src_tag, tgt_tag, os.path.join(TOK_TEST_DIR, f"{split}_{code}"))


def main():
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = IndicTransTokenizerEngine(MODEL_NAME, token=hf_token)
    process_train(tokenizer)
    process_test(tokenizer)

    print("\nDone. Tokenized files saved in:")
    print(f"  {TOK_TRAIN_DIR}")
    print(f"  {TOK_TEST_DIR}")


if __name__ == "__main__":
    main()