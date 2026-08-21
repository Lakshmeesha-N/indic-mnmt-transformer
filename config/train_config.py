"""
config/train_config.py

Configuration for the training stage.
Auto-detects Google Colab environment, mounts Google Drive if running on Colab,
and routes checkpoints and logs to Google Drive automatically without needing a .env file.
"""

import os
import sys
from pydantic_settings import BaseSettings, SettingsConfigDict


def is_colab() -> bool:
    """Checks if the code is executing inside a Google Colab notebook environment."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def mount_google_drive(mount_point: str = "/content/drive"):
    """
    Mounts Google Drive automatically if running in Google Colab.
    Safe to call multiple times (checks if already mounted).
    """
    if is_colab():
        try:
            from google.colab import drive
            if not os.path.exists(mount_point):
                print(f"[Drive] Mounting Google Drive to {mount_point} ...")
                drive.mount(mount_point)
            else:
                print(f"[Drive] Google Drive is already mounted at {mount_point}")
        except Exception as e:
            print(f"[Drive] Warning: Could not mount Google Drive ({e}). Falling back to local storage.")


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
    SAVE_EVERY: int = 1
    USE_AMP: bool = True

    # --- per-language BLEU / chrF++ evaluation ---
    BLEU_EVAL_EVERY: int = 1     # run greedy-decode eval every N epochs
    BLEU_MAX_SAMPLES: int = 200  # max val examples per language to decode

    # --- storage & google drive settings ---
    USE_DRIVE_ON_COLAB: bool = True
    DRIVE_FOLDER_NAME: str = "MMT"  # Folder created in your Google Drive

    # Optional manual overrides (if set, takes highest priority)
    CKPT_DIR_OVERRIDE: str = ""
    LOG_DIR_OVERRIDE: str = ""

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
        return os.path.join(self.BASE_DIR, "data", "tokenized", "train")

    @property
    def CKPT_DIR(self) -> str:
        if self.CKPT_DIR_OVERRIDE:
            return self.CKPT_DIR_OVERRIDE

        if is_colab() and self.USE_DRIVE_ON_COLAB:
            mount_google_drive()
            drive_ckpt_dir = f"/content/drive/MyDrive/{self.DRIVE_FOLDER_NAME}/checkpoints"
            os.makedirs(drive_ckpt_dir, exist_ok=True)
            return drive_ckpt_dir

        return os.path.join(self.BASE_DIR, "checkpoints")

    @property
    def LOG_DIR(self) -> str:
        if self.LOG_DIR_OVERRIDE:
            return self.LOG_DIR_OVERRIDE

        if is_colab() and self.USE_DRIVE_ON_COLAB:
            mount_google_drive()
            drive_log_dir = f"/content/drive/MyDrive/{self.DRIVE_FOLDER_NAME}/logs"
            os.makedirs(drive_log_dir, exist_ok=True)
            return drive_log_dir

        return os.path.join(self.BASE_DIR, "logs")

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