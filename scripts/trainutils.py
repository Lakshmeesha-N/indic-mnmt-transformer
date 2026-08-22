"""
scripts/train_utils.py

Training loop helpers, kept separate from train.py so train.py stays a thin
orchestrator. Includes:
  - Transformer warmup + inverse-sqrt LR schedule
  - run_epoch(): forward/backward pass with automatic mixed precision (AMP)
                 and a tqdm progress bar
  - save_checkpoint() / load_checkpoint()
"""

import os
import math
import torch
from tqdm import tqdm


def lr_lambda(step, warmup_steps, total_steps):
    """
    Warmup + Cosine Decay schedule.
    - Linear warmup from 0 to peak LR over `warmup_steps`.
    - Cosine decay from peak LR down to ~0 over the remaining steps, ending at `total_steps`.

    Note: total_steps must reflect your full intended training length. If you
    later extend training (e.g. train 10 epochs, then decide to add 2 more),
    update NUM_EPOCHS in config/train_config.py before resuming so total_steps
    is recalculated correctly for the new, longer run.
    """
    step = max(step, 1)
    if step < warmup_steps:
        return step / warmup_steps

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(model, dataloader, optimizer, scheduler, loss_fn, scaler,
              device, pad_token_id, epoch, use_amp=True, is_train=True, log_every=100,
              step_log_path=None):
    """
    Runs one full pass over dataloader (train or validation).

    Args:
        scaler: torch.cuda.amp.GradScaler (used only when is_train and use_amp)
        use_amp: whether to use mixed precision (autocast)
    Returns:
        avg_loss, perplexity
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(dataloader, desc=f"{'train' if is_train else 'val'} epoch {epoch}", leave=False)

    with torch.set_grad_enabled(is_train):
        for step, (src_ids, tgt_ids) in enumerate(progress):
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            decoder_input = tgt_ids[:, :-1]
            labels = tgt_ids[:, 1:]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(src_ids, decoder_input)  # [batch, tgt_len-1, vocab_size]
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()

            non_pad = (labels != pad_token_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

            if is_train and step % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                step_loss = loss.item()
                step_ppl = math.exp(min(step_loss, 20))
                progress.set_postfix(loss=f"{step_loss:.4f}", lr=f"{current_lr:.2e}")
                if step_log_path is not None:
                    log_step_row(step_log_path, epoch, step, step_loss, step_ppl, current_lr)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def save_checkpoint(path, model, optimizer, epoch, val_loss, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ---------------------------------------------------------------------------
# CSV logging helpers
# ---------------------------------------------------------------------------

def init_csv_logs(log_dir):
    """
    Creates the logs/ directory and initialises CSV files with headers.
    Safe to call on every run — skips header if the file already exists.

    Returns:
        (train_log_path, bleu_log_path, step_log_path)
    """
    import csv
    os.makedirs(log_dir, exist_ok=True)

    train_log = os.path.join(log_dir, "train_log.csv")
    bleu_log  = os.path.join(log_dir, "bleu_log.csv")
    step_log  = os.path.join(log_dir, "step_log.csv")

    if not os.path.exists(train_log):
        with open(train_log, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "language", "train_loss", "val_loss", "val_ppl", "timestamp"])

    if not os.path.exists(bleu_log):
        with open(bleu_log, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "language", "bleu_score", "chrf_score"])

    if not os.path.exists(step_log):
        with open(step_log, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss", "train_ppl", "lr", "timestamp"])

    return train_log, bleu_log, step_log


def log_step_row(path, epoch, step, train_loss, train_ppl, lr):
    """Appends one step row to step_log.csv every N steps."""
    import csv
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([epoch, step, f"{train_loss:.6f}", f"{train_ppl:.4f}", f"{lr:.6e}", timestamp])


def log_train_row(path, epoch, lang, train_loss, val_loss, val_ppl):
    """Appends one row to train_log.csv."""
    import csv
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([epoch, lang, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_ppl:.4f}", timestamp])


def log_bleu_row(path, epoch, lang, bleu, chrf):
    """Appends one row to bleu_log.csv."""
    import csv
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([epoch, lang, f"{bleu:.4f}", f"{chrf:.4f}"])


# ---------------------------------------------------------------------------
# Per-language validation — teacher-forced loss pass
# ---------------------------------------------------------------------------

def run_val_loss(model, val_dataset, loss_fn, device, pad_token_id, batch_size=32):
    """
    Teacher-forced forward pass over a single-language val dataset.
    Fast: no decoding, runs every epoch.

    Args:
        val_dataset: a MNMTDataset instance for one language
        loss_fn:     same CrossEntropyLoss used during training (ignore_index=pad)

    Returns:
        (avg_loss, perplexity)
    """
    from torch.utils.data import DataLoader
    from scripts.data_utils import Collator

    loader = DataLoader(val_dataset, batch_size=batch_size,
                        collate_fn=Collator(pad_token_id), shuffle=False)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src_ids, tgt_ids in loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            decoder_input = tgt_ids[:, :-1]
            labels        = tgt_ids[:, 1:]

            logits = model(src_ids, decoder_input)
            loss   = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            non_pad = (labels != pad_token_id).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


# ---------------------------------------------------------------------------
# Greedy decoding + BLEU / chrF++ evaluation
# ---------------------------------------------------------------------------

def greedy_decode(model, src_ids, tokenizer, device, max_len=128):
    """
    Autoregressive greedy decoding for a single source sequence.

    Args:
        src_ids:   1-D LongTensor [src_len] (already includes lang tags + eos)
        tokenizer: IndicTransTokenizerEngine instance — provides tgt_eos_id and
                   tgt_vocab (for bos token lookup)
        device:    torch device

    Returns:
        List[int] — predicted target token IDs (bos excluded, stops at eos exclusive)
    """
    model.eval()
    bos_id = tokenizer.tgt_vocab.get("<s>", 0)
    eos_id = tokenizer.tgt_eos_id

    src = src_ids.unsqueeze(0).to(device)           # [1, src_len]
    dec_input = torch.tensor([[bos_id]], device=device)  # [1, 1]

    predicted_ids = []
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src, dec_input)           # [1, cur_len, vocab_size]
            next_id = logits[0, -1, :].argmax(-1).item()
            if next_id == eos_id:
                break
            predicted_ids.append(next_id)
            dec_input = torch.cat(
                [dec_input, torch.tensor([[next_id]], device=device)], dim=1
            )

    return predicted_ids


def evaluate_bleu(model, val_dataset, tokenizer, device, max_samples=200, max_len=128,
                  lang_code="??", num_print=5):
    """
    Runs greedy decoding on up to max_samples pairs from val_dataset and
    computes corpus BLEU and chrF++ using sacrebleu.

    Also prints `num_print` side-by-side examples:
        SRC  : decoded English source
        REF  : ground-truth target text
        HYP  : model-generated target text

    Args:
        val_dataset: MNMTDataset for a single language
        tokenizer:   IndicTransTokenizerEngine — used for decoding predicted IDs
                     and reference IDs back to text via decode_tgt_batch()
        max_samples: cap on how many examples to decode (keeps eval fast)
        max_len:     maximum decoder steps per example
        lang_code:   language code string shown in the printed header (e.g. "hi")
        num_print:   how many examples to print (default 5)

    Returns:
        (bleu_score, chrf_score) — both as floats (0–100 scale)
    """
    import sacrebleu

    model.eval()
    hypotheses = []
    references = []
    src_texts   = []   # decoded source sentences for printing
    n = min(max_samples, len(val_dataset))

    for i in range(n):
        src_ids, tgt_ids = val_dataset[i]

        pred_ids = greedy_decode(model, src_ids, tokenizer, device, max_len)
        ref_ids  = tgt_ids.tolist()

        # Decode hypothesis and reference target text
        hyp = tokenizer.decode_tgt_batch([pred_ids], stop_at_eos=False)[0]
        ref = tokenizer.decode_tgt_batch([ref_ids],  stop_at_eos=True)[0]

        hypotheses.append(hyp)
        references.append(ref)

        # Decode source (English) for printing — skip lang-tag IDs at the front
        if i < num_print:
            src_text = tokenizer.decode_src_batch([src_ids.tolist()], stop_at_eos=True)[0]
            src_texts.append(src_text)

    # tokenize="none": we pass already-detokenized text — avoids double tokenization
    # which would give misleadingly wrong BLEU for Indic scripts
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="none").score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

    # Print side-by-side examples
    print(f"\n  --- Translation Examples [{lang_code}] (first {num_print}) ---")
    for i in range(min(num_print, len(hypotheses))):
        src_display = src_texts[i] if i < len(src_texts) else "(n/a)"
        print(f"  [{i+1}] SRC : {src_display}")
        print(f"       REF : {references[i]}")
        print(f"       HYP : {hypotheses[i] if hypotheses[i].strip() else '(empty)'}")
        print()

    return bleu, chrf