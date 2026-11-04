"""Chemins et configuration explicites du Trainer."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
GIT_ROOT = PACKAGE_ROOT.parent


def _env_file() -> Path:
    return GIT_ROOT / ".env"


class TrainerSettings(BaseSettings):
    """Variables d'environnement lues depuis le `.env` à la racine Git."""

    model_config = SettingsConfigDict(
        env_file=str(_env_file()),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    storage_root: Path = Field(
        default=GIT_ROOT / "pokecoach-trainer" / "storage",
        alias="TRAINER_STORAGE_ROOT",
    )
    vgc_bench_root: Path = Field(default=PACKAGE_ROOT / "vendor" / "vgc-bench")
    reference_checkpoint_path: Path | None = Field(default=None, alias="REFERENCE_CHECKPOINT_PATH")
    max_inbox_bytes: int = Field(default=100 * 1024 * 1024, alias="TRAINER_MAX_INBOX_BYTES")
    api_host: str = Field(default="127.0.0.1", alias="TRAINER_API_HOST")
    api_port: int = Field(default=8010, alias="TRAINER_API_PORT")


@lru_cache
def get_settings() -> TrainerSettings:
    return TrainerSettings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()


def storage_path(*parts: str) -> Path:
    root = get_settings().storage_root
    return root.joinpath(*parts)


def ensure_storage_layout() -> None:
    for name in (
        "inbox",
        "archive",
        "base_models",
        "staging",
        "raw_logs",
        "datasets",
        "jobs",
        "models",
        "runs",
    ):
        storage_path(name).mkdir(parents=True, exist_ok=True)


def load_dotenv_explicit() -> None:
    env = _env_file()
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
