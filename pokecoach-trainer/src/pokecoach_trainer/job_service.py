"""Soumission et exécution des jobs (collecte, préparation, entraînement)."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pokecoach_trainer.errors import Conflict
from pokecoach_trainer.jobs import (
    acquire_job_lock,
    get_job,
    release_job_lock,
    save_job,
    write_job_payload,
)
from pokecoach_trainer.schemas import (
    CollectionJobConfig,
    DatasetJobConfig,
    ExperimentConfig,
    JobConfig,
    JobKind,
    JobRecord,
    JobStatus,
    TrainingJobConfig,
)
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import new_opaque_id


def _experiment_from_record(record: JobRecord) -> ExperimentConfig:
    cfg = record.config
    if isinstance(cfg, ExperimentConfig):
        return cfg
    if isinstance(cfg, TrainingJobConfig):
        return cfg.experiment
    raise ValueError("job sans configuration d'entraînement")


def read_job_logs(job_id: str, tail: int = 100) -> list[str]:
    tail = max(1, min(tail, 1000))
    log_path = storage_path("jobs", job_id, "worker.log")
    if not log_path.is_file():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return lines[-tail:]


def submit_job(kind: JobKind, config: JobConfig | ExperimentConfig) -> JobRecord:
    """Réserve le slot, persiste l'état `queued` et lance le worker."""

    if kind == JobKind.training:
        from pokecoach_trainer.training_service import preflight

        experiment = (
            config
            if isinstance(config, ExperimentConfig)
            else config.experiment  # type: ignore[union-attr]
        )
        preflight(experiment)
        payload: JobConfig | ExperimentConfig = (
            config if isinstance(config, ExperimentConfig) else config
        )
    else:
        if not isinstance(
            config, (CollectionJobConfig, DatasetJobConfig, TrainingJobConfig)
        ):
            raise ValueError("configuration typée requise pour ce type de job")
        payload = config

    job_id = new_opaque_id("job-")
    run_dir = storage_path("runs", job_id)
    write_job_payload(job_id, payload)
    record = JobRecord(
        job_id=job_id,
        kind=kind,
        status=JobStatus.queued,
        config=payload,
        run_dir=str(run_dir),
    )
    save_job(record)
    try:
        acquire_job_lock(job_id)
    except RuntimeError as exc:
        record.status = JobStatus.failed
        record.error = str(exc)
        save_job(record)
        raise Conflict(str(exc), details={"job_id": job_id}) from exc

    log_path = storage_path("jobs", job_id, "worker.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parents[2]
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pokecoach_trainer.worker",
                    "--job-id",
                    job_id,
                    "--kind",
                    kind.value,
                ],
                cwd=str(package_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        record.status = JobStatus.running
        record.started_at = datetime.now(timezone.utc)
        record.pid = proc.pid
        save_job(record)
    except Exception as exc:
        record.status = JobStatus.failed
        record.error = str(exc)
        save_job(record)
        release_job_lock(job_id)
        raise
    return record


def run_job(job_id: str) -> None:
    """Point d'entrée worker : délègue selon le kind."""

    record = get_job(job_id)
    if record.kind == JobKind.training:
        from pokecoach_trainer.training_service import run_training_job

        run_training_job(job_id)
        return
    if record.kind == JobKind.collection:
        from pokecoach_trainer.collection_service import run_collection_job

        run_collection_job(job_id)
        return
    if record.kind == JobKind.prepare:
        from pokecoach_trainer.dataset_service import run_prepare_job

        run_prepare_job(job_id)
        return
    raise NotImplementedError(f"kind de job non implémenté: {record.kind.value}")
