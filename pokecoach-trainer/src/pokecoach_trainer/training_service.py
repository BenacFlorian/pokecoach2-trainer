"""Préflight et soumission de jobs d'entraînement."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pokecoach_trainer.base_model_service import (
    assert_dataset_contract_compatible,
    read_base_model_manifest,
)
from pokecoach_trainer.dataset_service import get_dataset
from pokecoach_trainer.errors import InvalidConfiguration, NotFound
from pokecoach_trainer.job_service import submit_job
from pokecoach_trainer.jobs import get_job, save_job
from pokecoach_trainer.schemas import DatasetState, ExperimentConfig, JobKind, JobRecord, JobStatus
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import write_json_atomic
from pokecoach_trainer.training import train_bc


def preflight(config: ExperimentConfig) -> None:
    """Vérifie dataset prêt, parent enregistré et compatibilité de contrat."""

    if config.initialization is None:
        raise InvalidConfiguration("initialization.base_model_id requis")
    try:
        manifest = get_dataset(config.dataset_id)
    except FileNotFoundError as exc:
        raise ValueError(f"dataset inconnu: {config.dataset_id}") from exc
    if manifest.state != DatasetState.ready:
        raise ValueError("dataset non prêt")
    if manifest.transitions_train < 1:
        raise ValueError("aucune transition train")
    try:
        parent = read_base_model_manifest(config.initialization.base_model_id)
    except NotFound as exc:
        raise InvalidConfiguration(exc.message) from exc
    assert_dataset_contract_compatible(parent, manifest)


def submit_training(config: ExperimentConfig) -> JobRecord:
    return submit_job(JobKind.training, config)


def run_training_job(job_id: str) -> JobRecord:
    from pokecoach_trainer.job_service import _experiment_from_record

    record = get_job(job_id)
    record.status = JobStatus.running
    save_job(record)
    config = _experiment_from_record(record)
    run_dir = Path(record.run_dir or storage_path("runs", job_id))
    log_path = storage_path("jobs", job_id, "worker.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = train_bc(config, run_dir)
        record.status = JobStatus.completed
        record.finished_at = datetime.now(timezone.utc)
        record.result_ref = str(run_dir / "result.json")
        write_json_atomic(run_dir / "result.json", result.model_dump(mode="json"))
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
