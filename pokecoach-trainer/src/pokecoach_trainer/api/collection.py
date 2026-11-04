"""API collecte Showdown."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from pokecoach_trainer.collection_service import submit_collection
from pokecoach_trainer.errors import Conflict, InvalidConfiguration
from pokecoach_trainer.job_service import read_job_logs
from pokecoach_trainer.jobs import get_job
from pokecoach_trainer.schemas import CollectionJobConfig, JobKind

router = APIRouter()


@router.post("/jobs", status_code=202)
def create_collection_job(body: CollectionJobConfig):
    try:
        return submit_collection(body.config)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    except InvalidConfiguration as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict()) from exc
    except RuntimeError as exc:
        if "occupé" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
def read_collection_job(job_id: str):
    try:
        record = get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if record.kind != JobKind.collection:
        raise HTTPException(status_code=404, detail="job de collecte introuvable")
    return record


@router.get("/jobs/{job_id}/logs")
def collection_job_logs(job_id: str, tail: int = 100):
    try:
        record = get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if record.kind != JobKind.collection:
        raise HTTPException(status_code=404, detail="job de collecte introuvable")
    lines = read_job_logs(job_id, tail=tail)
    return PlainTextResponse("\n".join(lines))
