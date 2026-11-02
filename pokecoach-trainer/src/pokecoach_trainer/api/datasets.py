"""API datasets."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from pokecoach_trainer.dataset_service import get_dataset, import_logs, prepare_dataset
from pokecoach_trainer.schemas import PrepareConfig

router = APIRouter()


@router.post("/import")
def api_import(body: dict):
    try:
        manifest = import_logs(body["relative_path"], body["dataset_id"])
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return manifest


@router.post("/{dataset_id}/prepare")
def api_prepare(dataset_id: str, body: PrepareConfig | None = None):
    body = body or PrepareConfig()
    try:
        return prepare_dataset(dataset_id, body)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{dataset_id}")
def api_get(dataset_id: str):
    try:
        return get_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
