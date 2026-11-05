"""API datasets (création, préparation asynchrone, inspection)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from pokecoach_trainer.dataset_service import (
    create_dataset,
    get_dataset,
    import_logs,
    list_rejections,
    prepare_dataset,
    reserve_legacy_import_dataset,
    submit_preparation,
)
from pokecoach_trainer.errors import Conflict, InvalidConfiguration, NotFound, TrainerError
from pokecoach_trainer.job_service import read_job_logs
from pokecoach_trainer.jobs import get_job
from pokecoach_trainer.schemas import (
    CreateDatasetRequest,
    DatasetState,
    JobKind,
    LegacyImportDatasetRequest,
    PrepareConfig,
    PrepareExecutionConfig,
)

router = APIRouter()


def _trainer_http(exc: TrainerError) -> HTTPException:
    status = 422
    if isinstance(exc, NotFound):
        status = 404
    elif isinstance(exc, Conflict):
        status = 409
    return HTTPException(status_code=status, detail=exc.to_dict())


@router.post("")
def api_create_dataset(body: CreateDatasetRequest):
    try:
        return create_dataset(
            body.snapshot_id,
            body.recipe,
            dataset_id=body.dataset_id,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, InvalidConfiguration) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/import", status_code=202)
def api_import_legacy(body: LegacyImportDatasetRequest):
    try:
        manifest = reserve_legacy_import_dataset(
            body.inbox_relpath,
            body.recipe,
            dataset_id=body.dataset_id,
        )
        job = submit_preparation(manifest.dataset_id, execution=PrepareExecutionConfig())
        return {"dataset": manifest, "job": job}
    except Conflict as exc:
        raise _trainer_http(exc) from exc
    except (ValueError, InvalidConfiguration, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/import-legacy")
def api_import(body: dict):
    """Import historique synchrone vers raw_logs (compatibilité palier 1)."""

    try:
        manifest = import_logs(body["relative_path"], body["dataset_id"])
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return manifest


@router.post("/{dataset_id}/prepare", status_code=202)
def api_prepare(
    dataset_id: str,
    body: PrepareExecutionConfig | PrepareConfig | None = None,
):
    manifest_path_exists = True
    try:
        existing = get_dataset(dataset_id)
    except FileNotFoundError:
        manifest_path_exists = False
        existing = None

    if existing and existing.state == DatasetState.imported:
        if isinstance(body, PrepareExecutionConfig):
            raise HTTPException(
                status_code=422,
                detail="corps PrepareExecutionConfig incompatible avec import legacy",
            )
        legacy_body = body if isinstance(body, PrepareConfig) else PrepareConfig()
        try:
            return prepare_dataset(dataset_id, legacy_body)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not manifest_path_exists:
        raise HTTPException(status_code=404, detail="dataset introuvable")
    if isinstance(body, PrepareConfig):
        raise HTTPException(
            status_code=422,
            detail="corps PrepareConfig incompatible avec snapshot ; utiliser conversion_workers",
        )
    execution = (
        body
        if isinstance(body, PrepareExecutionConfig)
        else PrepareExecutionConfig()
    )
    try:
        job = submit_preparation(dataset_id, execution=execution)
        return {"dataset_id": dataset_id, "job": job}
    except Conflict as exc:
        raise _trainer_http(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{dataset_id}")
def api_get(dataset_id: str):
    try:
        return get_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{dataset_id}/rejections")
def api_rejections(
    dataset_id: str,
    offset: int = 0,
    limit: int = 100,
    stage: str | None = None,
    reason_code: str | None = None,
):
    try:
        items, total = list_rejections(
            dataset_id,
            offset=offset,
            limit=limit,
            stage=stage,
            reason_code=reason_code,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"items": items, "total": total, "offset": offset, "limit": limit}


@router.get("/{dataset_id}/logs")
def api_logs(dataset_id: str, tail: int = 100):
    try:
        manifest = get_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not manifest.prepare_job_id:
        return PlainTextResponse("")
    try:
        record = get_job(manifest.prepare_job_id)
    except FileNotFoundError:
        return PlainTextResponse("")
    if record.kind != JobKind.prepare:
        raise HTTPException(status_code=404, detail="job prepare introuvable")
    lines = read_job_logs(manifest.prepare_job_id, tail=tail)
    return PlainTextResponse("\n".join(lines))
