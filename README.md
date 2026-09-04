# Multilingual Neural Machine Translation (English → Hindi, Kannada, Tamil)

A from-scratch Transformer-based Multilingual Neural Machine Translation (MNMT) system that translates English into three Indian languages — **Hindi**, **Kannada**, and **Tamil** — using a single shared model.

## Objective

Build and evaluate an MNMT system, and investigate whether **cross-lingual transfer** benefits low-resource language pairs: does jointly training one shared model on English→Hindi, English→Kannada, and English→Tamil simultaneously improve translation quality for the lower-resource languages (Kannada, Tamil), compared to training each pair separately? Shared subword and representation learning across languages is compared against per-language baselines to measure this effect.

## Architecture

A custom encoder-decoder Transformer, implemented from scratch in PyTorch, using modern architectural choices rather than the original 2017 design:

- **Rotary Position Embeddings (RoPE)** instead of sinusoidal positional encoding
- **RMSNorm** instead of LayerNorm
- **Pre-norm residual connections**
- **AdamW optimizer** with warmup + cosine learning rate decay
- An optional **split / multi-expert feed-forward layer** (novel low-resource technique), toggleable against a standard baseline FFN via config, for direct baseline-vs-novel comparison

## Tokenization

Reuses **IndicTrans2's** pretrained SentencePiece models and vocabularies (separate source/target tokenizers, `dict.SRC.json` / `dict.TGT.json`), rather than training a new tokenizer from scratch.

## Data

- **Training:** [AI4Bharat Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) parallel corpus (English-Hindi, English-Kannada, English-Tamil)
- **Evaluation:** [IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen) and [IN22-Conv](https://huggingface.co/datasets/ai4bharat/IN22-Conv) benchmark test sets

## Evaluation Metrics

BLEU, chrF++, TER, COMET, METEOR, and ROUGE-L — computed **per language** (Hindi, Kannada, Tamil), comparing the baseline model against the multilingual/novel-technique variant.

## Project Structure

```
config/               # environment & hyperparameter configs
├── download_config.py
├── preprocess_config.py
└── train_config.py

models/                # Transformer architecture (built from scratch)
├── config.py           # central model hyperparameter config
├── tokenizer.py         # IndicTrans2 tokenizer wrapper (encode/decode)
├── embeddings.py
├── rope.py
├── norm.py
├── residual.py
├── attention.py
├── masks.py
├── feedforward.py
├── moe_feedforward.py   # split/multi-expert FFN (novel technique)
├── encoder.py
├── decoder.py
└── transformer.py       # full assembled model

scripts/                # data pipeline + training helpers
├── download_data.py
├── preprocess.py
├── data_utils.py         # Dataset, Collator
└── train_utils.py        # LR schedule, run_epoch, checkpointing

data/                    # created by scripts, not committed to git
├── raw/train, raw/test
└── tokenized/train, tokenized/test

checkpoints/              # saved model weights
logs/                      # training logs (loss, perplexity, BLEU per language)
train.py                    # training entry point
```

## Setup

**1. Clone and create a virtual environment**
```bash
git clone <your-repo-url>
cd MMT
python -m venv mmt_env
mmt_env\Scripts\activate        # Windows
# source mmt_env/bin/activate   # Mac/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up HuggingFace access**

IndicTrans2 and IN22-Gen/IN22-Conv are gated datasets/models on HuggingFace.
- Create a free account at https://huggingface.co
- Accept access conditions on:
  - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
  - https://huggingface.co/datasets/ai4bharat/IN22-Gen
  - https://huggingface.co/datasets/ai4bharat/IN22-Conv
- Create a **Read** token at https://huggingface.co/settings/tokens
- Create a `.env` file in the project root:
```
HF_TOKEN=hf_your_token_here
```

## Usage

Run the full pipeline in order:

**1. Download data**
```bash
python scripts/download_data.py
```
Downloads Samanantar (train) and IN22-Gen/IN22-Conv (test) into `data/raw/`.

**2. Preprocess**
```bash
python scripts/preprocess.py
```
Cleans, tags, and tokenizes the data into `data/tokenized/`.

**3. Train**
```bash
python train.py
```
Trains the model, logging loss/perplexity/BLEU per language to `logs/`, and saving checkpoints to `checkpoints/`. Automatically resumes from the latest checkpoint if one exists.



## References

- [AI4Bharat IndicTrans2](https://github.com/ai4bharat/IndicTrans2)
- [ACL Anthology](https://aclanthology.org/)
- [2024 NAACL Findings paper](https://aclanthology.org/2024.findings-naacl.176.pdf)
- [2024 EAMT paper](https://aclanthology.org/2024.eamt-1.19.pdf)