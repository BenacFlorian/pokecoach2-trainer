"""Journaux opérateur pour campagnes de collecte (flush immédiat)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokecoach_trainer.settings import storage_path


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def campaign_log_path(campaign_id: str) -> Path:
    return storage_path("jobs", campaign_id, "worker.log")


def worker_log_path(campaign_id: str, worker_id: str) -> Path:
    return storage_path("jobs", campaign_id, "workers", f"{worker_id}.log")


def log_campaign(campaign_id: str, message: str) -> None:
    _append(campaign_log_path(campaign_id), f"{_ts()} [supervisor] {message}\n")


def log_worker(campaign_id: str, worker_id: str, message: str) -> None:
    _append(worker_log_path(campaign_id, worker_id), f"{_ts()} [{worker_id}] {message}\n")


def format_progress_snapshot(progress: dict[str, Any]) -> str:
    counters = progress.get("counters") or {}
    parts = [
        f"phase={progress.get('phase')}",
        f"selected={counters.get('selected_replays', 0)}",
        f"new={counters.get('new_published', 0)}",
        f"requests={counters.get('requests_consumed', 0)}",
        f"pending_dl={counters.get('pending_downloads', 0)}",
        f"windows_open={progress.get('discovery_windows_open', 0)}",
    ]
    rejected = progress.get("rejected_by_reason") or {}
    if rejected:
        rej = ",".join(f"{k}:{v}" for k, v in sorted(rejected.items()))
        parts.append(f"rejected=({rej})")
    return " ".join(parts)
