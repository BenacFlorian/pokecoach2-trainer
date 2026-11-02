"""État persistant des jobs et verrou d'entraînement."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pokecoach_trainer.schemas import ExperimentConfig, JobRecord, JobStatus
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import validate_id, write_json_atomic


def _job_dir(job_id: str) -> Path:
    return storage_path("jobs", validate_id(job_id))


def _state_path(job_id: str) -> Path:
    return _job_dir(job_id) / "state.json"


def _lock_path() -> Path:
    return storage_path("jobs", ".training.lock")


def save_job(record: JobRecord) -> None:
    directory = _job_dir(record.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(_state_path(record.job_id), record.model_dump(mode="json"))


def get_job(job_id: str) -> JobRecord:
    return JobRecord.model_validate(json.loads(_state_path(job_id).read_text(encoding="utf-8")))


def acquire_training_lock(job_id: str) -> None:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, job_id.encode())
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError("slot d'entraînement occupé") from exc


def release_training_lock(job_id: str) -> None:
    path = _lock_path()
    if path.is_file():
        content = path.read_text(encoding="utf-8").strip()
        if content == job_id:
            path.unlink(missing_ok=True)


def reconcile_jobs() -> None:
    """Marque les jobs orphelins comme interrompus au redémarrage."""

    jobs_root = storage_path("jobs")
    if not jobs_root.is_dir():
        return
    for state_file in jobs_root.glob("*/state.json"):
        record = JobRecord.model_validate(json.loads(state_file.read_text(encoding="utf-8")))
        if record.status in {JobStatus.running, JobStatus.queued}:
            record.status = JobStatus.interrupted
            record.finished_at = datetime.now(timezone.utc)
            record.error = record.error or "interrupted at reconcile"
            save_job(record)
    _lock_path().unlink(missing_ok=True)


def write_job_config(job_id: str, config: ExperimentConfig) -> Path:
    directory = _job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    write_json_atomic(path, config.model_dump(mode="json"))
    return path
