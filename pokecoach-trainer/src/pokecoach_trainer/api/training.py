"""API entraînement."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from pokecoach_trainer.jobs import get_job
from pokecoach_trainer.schemas import ExperimentConfig
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.training_service import submit_training

router = APIRouter()


@router.post("/jobs")
def create_job(config: ExperimentConfig):
    try:
        record = submit_training(config)
    except RuntimeError as exc:
        if "occupé" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return record


@router.get("/jobs/{job_id}")
def read_job(job_id: str):
    try:
        return get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/logs")
def job_logs(job_id: str, tail: int = 100):
    tail = max(1, min(tail, 1000))
    log_path = storage_path("jobs", job_id, "worker.log")
    if not log_path.is_file():
        return PlainTextResponse("")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return PlainTextResponse("\n".join(lines[-tail:]))
