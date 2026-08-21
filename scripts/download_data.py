"""
download_data.py

Purpose: DOWNLOAD ONLY. No cleaning, no filtering, no dedup here.
All cleaning happens later in preprocess.py.

Downloads:
  - Training data : ai4bharat/samanantar        -> configs: hi, kn, ta
  - Test data      : ai4bharat/IN22-Gen          -> config: "default"
                    : ai4bharat/IN22-Conv         -> config: "default"

Mode control (set in .env, see config.py):
  DOWNLOAD_MODE=develop   -> only downloads DEV_SAMPLE_SIZE pairs per language
  DOWNLOAD_MODE=train     -> downloads full datasets (default)

NOTE: IN22-Gen / IN22-Conv are gated datasets on HuggingFace.
Before running this script:
  1. Create a free account at https://huggingface.co
  2. Accept access conditions on the dataset pages.
  3. Set HF_TOKEN in .env. Do NOT hardcode/commit real tokens.

Output: plain TSV files saved into
  data/train/  (Samanantar training pairs)
  data/test/   (IN22-Gen / IN22-Conv benchmark pairs)
"""

import csv
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import download_settings as settings

settings.apply_hf_env()  

from datasets import load_dataset
from tqdm import tqdm

os.makedirs(settings.RAW_DIR, exist_ok=True)
os.makedirs(settings.TRAIN_DIR, exist_ok=True)
os.makedirs(settings.TEST_DIR, exist_ok=True)

MODE = settings.DOWNLOAD_MODE.strip().lower()

LANGS = {
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
}

META_COLS = ["domain", "bucket", "source"]


def download_samanantar():
    print(f"Downloading Samanantar training data (hi, kn, ta) [mode={MODE}]...")
    for code in LANGS:
        print(f"\n  loading ai4bharat/samanantar [{code}] ...")

        ds = None
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                if MODE == "develop":
                    ds = load_dataset("ai4bharat/samanantar", code, split="train", streaming=True)
                    ds = ds.take(settings.DEV_SAMPLE_SIZE)
                else:
                    ds = load_dataset("ai4bharat/samanantar", code, split="train")
                break
            except Exception as e:
                print(f"  [warn] attempt {attempt}/{settings.MAX_RETRIES} failed: {e}")
                if attempt == settings.MAX_RETRIES:
                    raise
                time.sleep(5 * attempt)

        out_path = os.path.join(settings.TRAIN_DIR, f"samanantar_{code}.tsv")
        count = 0
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["src", "tgt"])
            for row in tqdm(ds, desc=f"  samanantar_{code}", unit=" pairs"):
                writer.writerow([row["src"], row["tgt"]])
                count += 1

        print(f"  saved {out_path} ({count} pairs)")


def download_in22():
    print(f"\nDownloading IN22-Gen and IN22-Conv benchmark test sets [mode={MODE}]...")
    for hf_name in ["ai4bharat/IN22-Gen", "ai4bharat/IN22-Conv"]:
        short = hf_name.split("/")[-1]
        print(f"  loading {hf_name} [default] ...")

        ds_dict = None
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                ds_dict = load_dataset(hf_name, "default", token=settings.HF_TOKEN or None)
                break
            except Exception as e:
                print(f"  [warn] attempt {attempt}/{settings.MAX_RETRIES} failed: {e}")
                if attempt == settings.MAX_RETRIES:
                    raise
                time.sleep(5 * attempt)

        split_name = list(ds_dict.keys())[0]
        df_full = ds_dict[split_name].to_pandas()

        for code, script in LANGS.items():
            cols = ["eng_Latn", script] + [c for c in META_COLS if c in df_full.columns]
            df = df_full[cols].rename(columns={"eng_Latn": "src", script: "tgt"})

            if MODE == "develop":
                df = df.head(settings.DEV_SAMPLE_SIZE)

            out_path = os.path.join(settings.TEST_DIR, f"{short}_{code}.tsv")
            df.to_csv(out_path, sep="\t", index=False)
            print(f"  saved {out_path}  ({len(df)} pairs)")


if __name__ == "__main__":
    download_samanantar()
    download_in22()
    print(f"\nDone.")
    print(f"  Training data -> {settings.TRAIN_DIR}")
    print(f"  Test data     -> {settings.TEST_DIR}")
    print("Next step: run preprocess.py to clean, tag, and tokenize this data.")