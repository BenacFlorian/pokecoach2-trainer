"""API modèles publiés."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from pokecoach_trainer.artifact_service import list_models
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import read_json

router = APIRouter()


@router.get("")
def api_list():
    return list_models()


@router.get("/{model_id}")
def api_get(model_id: str):
    pub = storage_path("models", model_id, "published.json")
    if not pub.is_file():
        raise HTTPException(status_code=404, detail="modèle absent")
    return read_json(pub)


@router.get("/{model_id}/download")
def api_download(model_id: str):
    archive = storage_path("models", model_id, f"{model_id}.zip")
    if not archive.is_file():
        raise HTTPException(status_code=404, detail="package absent")
    return FileResponse(path=str(archive), filename=archive.name)
