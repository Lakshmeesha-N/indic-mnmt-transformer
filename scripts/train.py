"""
scripts/train.py

Training orchestrator for Multilingual Neural Machine Translation (MNMT):
  1. Configures environment and hardware device.
  2. Builds dataset: 98% combined training data and 2% per-language validation splits.
  3. Initializes Transformer model, optimizer, warmup+cosine scheduler, and AMP scaler.
  4. Automatically detects and resumes from existing checkpoints in checkpoints/ directory.
  5. Runs epoch training loop with teacher-forced loss/PPL per language.
  6. Periodically evaluates autoregressive greedy decoding (BLEU and chrF++) per language.
  7. Logs all metrics to structured CSV files (logs/train_log.csv and logs/bleu_log.csv).
  8. Saves periodic and best checkpoints.
"""

import os
import re
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.config import mnmt_config
from models.transformer import Transformer
from models.tokenizer import IndicTransTokenizerEngine
from config.train_config import train_settings as cfg
from scripts.data_utils import MNMTDataset, Collator
from scripts.trainutils import (
    lr_lambda,
    run_epoch,
    save_checkpoint,
    load_checkpoint,
    init_csv_logs,
    log_train_row,
    log_bleu_row,
    run_val_loss,
    evaluate_bleu,
)


def find_latest_checkpoint(ckpt_dir: str):
    """
    Scans ckpt_dir for saved checkpoints and returns the path to the most recent one.
    Prefers model_epoch{N}.pt with highest N, falls back to model_best.pt or any .pt file.
    """
    if not os.path.exists(ckpt_dir):
        return None

    epoch_ckpts = []
    for path in glob.glob(os.path.join(ckpt_dir, "model_epoch*.pt")):
        match = re.search(r"model_epoch(\d+)\.pt", os.path.basename(path))
        if match:
            epoch_ckpts.append((int(match.group(1)), path))

    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda x: x[0], reverse=True)
        return epoch_ckpts[0][1]

    best_path = os.path.join(ckpt_dir, "model_best.pt")
    if os.path.exists(best_path):
        return best_path

    other_ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if other_ckpts:
        other_ckpts.sort(key=os.path.getmtime, reverse=True)
        return other_ckpts[0]

    return None


def main():
    device = cfg.DEVICE
    print(f"Using device: {device}")

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # 1. Initialize CSV logs
    train_log_path, bleu_log_path = init_csv_logs(cfg.LOG_DIR)
    print(f"Logs will be written to:\n  {train_log_path}\n  {bleu_log_path}")

    # 2. Load tokenized dataset (98% train, 2% per-language val)
    print("\n--- Loading Data ---")
    train_dataset = MNMTDataset(
        cfg.TOK_TRAIN_DIR,
        val_fraction=cfg.VAL_FRACTION,
        is_train=True,
    )
    val_splits = MNMTDataset.per_language_splits(
        cfg.TOK_TRAIN_DIR,
        val_fraction=cfg.VAL_FRACTION,
    )

    collator = Collator(pad_token_id=mnmt_config.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    # 3. Initialize Model, Optimizer, Loss
    print("\n--- Initializing Model ---")
    model = Transformer(mnmt_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.NUM_EPOCHS
    print(f"Steps per epoch: {steps_per_epoch} | Total scheduled steps: {total_steps}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, cfg.WARMUP_STEPS, total_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and device.type == "cuda")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=mnmt_config.pad_token_id,
        label_smoothing=mnmt_config.label_smoothing,
    )

    # 4. Check for existing checkpoint to resume automatically
    start_epoch = 1
    best_val_loss = float("inf")
    latest_ckpt = find_latest_checkpoint(cfg.CKPT_DIR)

    if latest_ckpt:
        print(f"\n[RESUME] Found existing checkpoint: {latest_ckpt}")
        ckpt_data = load_checkpoint(latest_ckpt, model, optimizer, device=device)
        start_epoch = ckpt_data.get("epoch", 0) + 1
        best_val_loss = ckpt_data.get("val_loss", float("inf"))
        
        # Fast-forward learning rate scheduler to current training step
        recovered_step = (start_epoch - 1) * steps_per_epoch
        for _ in range(recovered_step):
            scheduler.step()
        print(f"Resuming training from epoch {start_epoch} (best previous val_loss={best_val_loss:.4f})")
    else:
        print("\nNo existing checkpoint found. Starting fresh training from Epoch 1.")

    if start_epoch > cfg.NUM_EPOCHS:
        print(f"Training already completed up to epoch {start_epoch - 1} (NUM_EPOCHS={cfg.NUM_EPOCHS}). Exiting.")
        return

    # 5. Initialize Tokenizer for BLEU / chrF++ evaluation
    print("\n--- Loading Tokenizer for Evaluation ---")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = IndicTransTokenizerEngine(token=hf_token)

    # 6. Main Training Loop
    print("\n--- Starting Training Loop ---")
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch} / {cfg.NUM_EPOCHS} ===")

        # Training pass (merged multilingually)
        train_loss, train_ppl = run_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, scaler,
            device, mnmt_config.pad_token_id, epoch,
            use_amp=cfg.USE_AMP, is_train=True, log_every=cfg.LOG_EVERY,
        )
        print(f"Train: loss={train_loss:.4f} | ppl={train_ppl:.2f}")

        # Per-language validation pass (loss + perplexity)
        val_losses = []
        for lang, val_ds in val_splits.items():
            v_loss, v_ppl = run_val_loss(
                model, val_ds, loss_fn, device, mnmt_config.pad_token_id, batch_size=cfg.BATCH_SIZE
            )
            val_losses.append(v_loss)
            log_train_row(train_log_path, epoch, lang, train_loss, v_loss, v_ppl)
            print(f"  Val [{lang}]: loss={v_loss:.4f} | ppl={v_ppl:.2f}")

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        # Periodic BLEU / chrF++ evaluation (greedy decoding per language)
        if epoch % cfg.BLEU_EVAL_EVERY == 0:
            print(f"  Evaluating BLEU & chrF++ (samples={cfg.BLEU_MAX_SAMPLES})...")
            for lang, val_ds in val_splits.items():
                bleu, chrf = evaluate_bleu(
                    model, val_ds, tokenizer, device,
                    max_samples=cfg.BLEU_MAX_SAMPLES,
                )
                log_bleu_row(bleu_log_path, epoch, lang, bleu, chrf)
                print(f"  Metrics [{lang}]: BLEU={bleu:.2f} | chrF++={chrf:.2f}")

        # Save periodic checkpoint
        if epoch % cfg.SAVE_EVERY == 0:
            ckpt_path = os.path.join(cfg.CKPT_DIR, f"model_epoch{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, avg_val_loss, mnmt_config)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(cfg.CKPT_DIR, "model_best.pt")
            save_checkpoint(best_path, model, optimizer, epoch, avg_val_loss, mnmt_config)
            print(f"  ★ New best model saved: {best_path} (avg_val_loss={avg_val_loss:.4f})")

    print("\n--- Training Complete ---")


if __name__ == "__main__":
    main()