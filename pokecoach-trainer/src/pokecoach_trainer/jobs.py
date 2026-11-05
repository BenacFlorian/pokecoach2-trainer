"""État persistant des jobs et verrous inter-processus (fcntl)."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pokecoach_trainer.schemas import ExperimentConfig, JobKind, JobRecord, JobStatus
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import validate_id, write_json_atomic

_compute_lock_fd: int | None = None


def _job_dir(job_id: str) -> Path:
    return storage_path("jobs", validate_id(job_id))


def _state_path(job_id: str) -> Path:
    return _job_dir(job_id) / "state.json"


def _lock_path(resource: Literal["collection", "compute"]) -> Path:
    name = ".collection.lock" if resource == "collection" else ".compute.lock"
    return storage_path("jobs", name)


def save_job(record: JobRecord) -> None:
    directory = _job_dir(record.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(_state_path(record.job_id), record.model_dump(mode="json"))


def get_job(job_id: str) -> JobRecord:
    path = _state_path(job_id)
    if not path.is_file():
        raise FileNotFoundError(job_id)
    return JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))


def acquire_resource_lock(resource: Literal["collection", "compute"], job_id: str) -> int:
    """Réserve un slot via flock sur .collection.lock ou .compute.lock."""

    global _compute_lock_fd
    path = _lock_path(resource)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError("slot de job long occupé") from exc
    os.ftruncate(fd, 0)
    os.write(fd, job_id.encode())
    os.fsync(fd)
    if resource == "compute":
        _compute_lock_fd = fd
    return fd


def release_resource_lock(fd: int) -> None:
    global _compute_lock_fd
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
        if fd == _compute_lock_fd:
            _compute_lock_fd = None


def acquire_job_lock(job_id: str) -> int:
    return acquire_resource_lock("compute", job_id)


def release_job_lock(job_id: str, fd: int | None = None) -> None:
    global _compute_lock_fd
    use_fd = fd if fd is not None else _compute_lock_fd
    if use_fd is None:
        return
    release_resource_lock(use_fd)


def acquire_training_lock(job_id: str) -> int:
    return acquire_resource_lock("compute", job_id)


def release_training_lock(job_id: str) -> None:
    release_job_lock(job_id)


def reconcile_jobs() -> None:
    """Marque les jobs JSON orphelins ; ne touche pas aux campagnes SQLite actives."""

    jobs_root = storage_path("jobs")
    if not jobs_root.is_dir():
        return
    for state_file in jobs_root.glob("*/state.json"):
        record = JobRecord.model_validate(json.loads(state_file.read_text(encoding="utf-8")))
        if record.kind == JobKind.collection:
            continue
        if record.status in {JobStatus.running, JobStatus.queued}:
            record.status = JobStatus.interrupted
            record.finished_at = datetime.now(timezone.utc)
            record.error = record.error or "interrupted at reconcile"
            save_job(record)
    if _compute_lock_fd is not None:
        release_resource_lock(_compute_lock_fd)


def write_job_config(job_id: str, config: ExperimentConfig) -> Path:
    directory = _job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    write_json_atomic(path, config.model_dump(mode="json"))
    return path


def write_job_payload(job_id: str, payload: object) -> Path:
    directory = _job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if hasattr(payload, "model_dump"):
        write_json_atomic(path, payload.model_dump(mode="json"))
    else:
        write_json_atomic(path, payload)
    return path
