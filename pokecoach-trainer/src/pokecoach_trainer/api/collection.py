"""API collecte Showdown."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from pokecoach_trainer.archive_service import create_campaign_snapshot, list_campaign_replays
from pokecoach_trainer.collection_metrics import collection_progress
from pokecoach_trainer.collection_service import (
    pause_collection,
    resume_collection,
    retry_failed_collection,
    scale_collection,
    stop_collection,
    submit_collection,
)
from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.errors import Conflict, InvalidConfiguration, NotFound
from pokecoach_trainer.job_service import read_collection_worker_logs, read_job_logs
from pokecoach_trainer.jobs import get_job
from pokecoach_trainer.schemas import (
    CollectionJobConfig,
    JobKind,
    ResumeCollectionRequest,
    RetryFailedRequest,
    ScaleCollectionRequest,
)
from pokecoach_trainer.settings import get_settings

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
    store = CollectionStore(get_settings().storage_root)
    try:
        store.get_campaign(job_id)
        progress = collection_progress(store, job_id)
    except NotFound:
        progress = record.progress or {}
    else:
        record.progress = progress
    return record


@router.post("/jobs/{job_id}/pause", status_code=202)
def pause_job(job_id: str):
    _assert_collection(job_id)
    return pause_collection(job_id).job


@router.post("/jobs/{job_id}/stop", status_code=202)
def stop_job(job_id: str):
    _assert_collection(job_id)
    return stop_collection(job_id).job


@router.post("/jobs/{job_id}/resume", status_code=202)
def resume_job(job_id: str, body: ResumeCollectionRequest):
    _assert_collection(job_id)
    try:
        return resume_collection(job_id, body).job
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    except InvalidConfiguration as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict()) from exc


@router.post("/jobs/{job_id}/scale", status_code=202)
def scale_job(job_id: str, body: ScaleCollectionRequest):
    _assert_collection(job_id)
    return scale_collection(job_id, body).job


@router.post("/jobs/{job_id}/retry-failed", status_code=202)
def retry_job(job_id: str, body: RetryFailedRequest):
    _assert_collection(job_id)
    count = retry_failed_collection(job_id, body)
    return {"retried": count}


@router.get("/jobs/{job_id}/logs")
def collection_job_logs(job_id: str, tail: int = 100):
    _assert_collection(job_id)
    lines = read_job_logs(job_id, tail=tail)
    return PlainTextResponse("\n".join(lines))


@router.get("/jobs/{job_id}/replays")
def collection_job_replays(job_id: str, offset: int = 0, limit: int = 100):
    _assert_collection(job_id)
    try:
        refs, total = list_campaign_replays(job_id, offset=offset, limit=limit)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=exc.to_dict()) from exc
    return {
        "total": total,
        "offset": max(0, offset),
        "limit": max(1, min(limit, 1000)),
        "refs": [r.model_dump(mode="json") for r in refs],
    }


@router.post("/jobs/{job_id}/snapshot", status_code=200)
def collection_job_snapshot(job_id: str):
    _assert_collection(job_id)
    try:
        snapshot = create_campaign_snapshot(job_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=exc.to_dict()) from exc
    return snapshot


@router.get("/jobs/{job_id}/workers")
def collection_job_workers(job_id: str):
    _assert_collection(job_id)
    store = CollectionStore(get_settings().storage_root)
    try:
        workers = store.list_workers(job_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=exc.to_dict()) from exc
    return {"workers": workers}


@router.get("/jobs/{job_id}/workers/{worker_id}/logs")
def collection_worker_logs(job_id: str, worker_id: str, tail: int = 100):
    _assert_collection(job_id)
    lines = read_collection_worker_logs(job_id, worker_id, tail=tail)
    return PlainTextResponse("\n".join(lines))


def _assert_collection(job_id: str) -> None:
    try:
        record = get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if record.kind != JobKind.collection:
        raise HTTPException(status_code=404, detail="job de collecte introuvable")
