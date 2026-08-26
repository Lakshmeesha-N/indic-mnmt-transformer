"""
Configuration for the download stage only.
Reads DOWNLOAD_* and HF_* env vars from .env using pydantic BaseSettings.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class DownloadSettings(BaseSettings):
    DOWNLOAD_MODE: str = "train"          # "develop" or "train"
    DEV_SAMPLE_SIZE: int = 100000

    HF_HUB_DOWNLOAD_TIMEOUT: str = "120"
    HF_HUB_ENABLE_HF_XET: str = "0"
    HF_HUB_DISABLE_SYMLINKS_WARNING: str = "1"
    HF_TOKEN: str = ""

    MAX_RETRIES: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def BASE_DIR(self) -> str:
        # project root = one level above the config/ folder
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def RAW_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "raw")

    @property
    def TRAIN_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "raw", "train")

    @property
    def TEST_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "raw", "test")

    def apply_hf_env(self):
        os.environ["HF_HUB_ENABLE_HF_XET"] = self.HF_HUB_ENABLE_HF_XET
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = self.HF_HUB_DISABLE_SYMLINKS_WARNING
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = self.HF_HUB_DOWNLOAD_TIMEOUT


download_settings = DownloadSettings()