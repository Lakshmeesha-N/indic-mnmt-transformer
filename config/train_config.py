"""
config/train_config.py

Configuration for the training and preprocessing stage.
Uses simple configurable paths and environment variables (.env / system env).
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainSettings(BaseSettings):
    # --- training hyperparameters ---
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_STEPS: int = 4000
    VAL_FRACTION: float = 0.02

    # --- logging / checkpointing ---
    LOG_EVERY: int = 100
    SAVE_EVERY: int = 5
    USE_AMP: bool = True

    # --- per-language BLEU / chrF++ evaluation ---
    ENABLE_BLEU_EVAL: bool = False  # Set to True to enable autoregressive BLEU/chrF++ eval
    BLEU_EVAL_EVERY: int = 1        # run greedy-decode eval every N epochs (if enabled)
    BLEU_MAX_SAMPLES: int = 200     # max val examples per language to decode

    # --- Path Overrides (optional: set via .env or environment variable) ---
    CKPT_DIR_OVERRIDE: str = ""
    LOG_DIR_OVERRIDE: str = ""
    TOK_DIR_OVERRIDE: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def BASE_DIR(self) -> str:
        # project root = one level above config/
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def TOK_TRAIN_DIR(self) -> str:
        if self.TOK_DIR_OVERRIDE:
            path = os.path.join(self.TOK_DIR_OVERRIDE, "train")
            os.makedirs(path, exist_ok=True)
            return path

        path = os.path.join(self.BASE_DIR, "data", "tokenized", "train")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def TOK_TEST_DIR(self) -> str:
        if self.TOK_DIR_OVERRIDE:
            path = os.path.join(self.TOK_DIR_OVERRIDE, "test")
            os.makedirs(path, exist_ok=True)
            return path

        path = os.path.join(self.BASE_DIR, "data", "tokenized", "test")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def CKPT_DIR(self) -> str:
        if self.CKPT_DIR_OVERRIDE:
            os.makedirs(self.CKPT_DIR_OVERRIDE, exist_ok=True)
            return self.CKPT_DIR_OVERRIDE

        path = os.path.join(self.BASE_DIR, "checkpoints")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def LOG_DIR(self) -> str:
        if self.LOG_DIR_OVERRIDE:
            os.makedirs(self.LOG_DIR_OVERRIDE, exist_ok=True)
            return self.LOG_DIR_OVERRIDE

        path = os.path.join(self.BASE_DIR, "logs")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def DEVICE(self):
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[Device] GPU detected: {device_name} (CUDA is available)")
            return torch.device("cuda")
        print("[Device] CUDA not available, using CPU.")
        return torch.device("cpu")


train_settings = TrainSettings()