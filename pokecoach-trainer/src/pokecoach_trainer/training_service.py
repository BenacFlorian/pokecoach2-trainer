"""Préflight et soumission de jobs d'entraînement."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pokecoach_trainer.artifact_service import export_model
from pokecoach_trainer.dataset_service import get_dataset
from pokecoach_trainer.jobs import (
    acquire_training_lock,
    get_job,
    save_job,
    write_job_config,
)
from pokecoach_trainer.schemas import DatasetState, ExperimentConfig, JobRecord, JobStatus
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import new_opaque_id
from pokecoach_trainer.training import train_bc


def preflight(config: ExperimentConfig) -> None:
    manifest = get_dataset(config.dataset_id)
    if manifest.state != DatasetState.ready:
        raise ValueError("dataset non prêt")
    if manifest.transitions_train < 1:
        raise ValueError("aucune transition train")


def submit_training(config: ExperimentConfig) -> JobRecord:
    preflight(config)
    job_id = new_opaque_id("job-")
    run_dir = storage_path("runs", job_id)
    write_job_config(job_id, config)
    record = JobRecord(job_id=job_id, status=JobStatus.queued, config=config, run_dir=str(run_dir))
    save_job(record)
    acquire_training_lock(job_id)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pokecoach_trainer.worker", "--job-id", job_id],
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        record.status = JobStatus.running
        record.started_at = datetime.now(timezone.utc)
        record.pid = proc.pid
        save_job(record)
    except Exception as exc:
        record.status = JobStatus.failed
        record.error = str(exc)
        save_job(record)
        from pokecoach_trainer.jobs import release_training_lock

        release_training_lock(job_id)
        raise
    return record


def run_training_job(job_id: str) -> JobRecord:
    record = get_job(job_id)
    record.status = JobStatus.running
    save_job(record)
    config = record.config
    run_dir = Path(record.run_dir or storage_path("runs", job_id))
    log_path = storage_path("jobs", job_id, "worker.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = train_bc(config, run_dir)
        export_model(result, config, model_id=job_id)
        record.status = JobStatus.completed
        record.finished_at = datetime.now(timezone.utc)
    except Exception as exc:
        record.status = JobStatus.failed
        record.error = str(exc)
        record.finished_at = datetime.now(timezone.utc)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(str(exc) + "\n")
        raise
    finally:
        from pokecoach_trainer.jobs import release_training_lock

        release_training_lock(job_id)
    save_job(record)
    return record
