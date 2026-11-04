"""API archive brute et snapshots."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query

from pokecoach_trainer.archive_service import create_snapshot, list_replays, read_snapshot
from pokecoach_trainer.schemas import SnapshotRequest

router = APIRouter()


@router.get("/replays")
def archive_replays(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    format_id: str | None = None,
    upload_after: datetime | None = None,
    upload_before: datetime | None = None,
):
    items, total = list_replays(
        offset=offset,
        limit=limit,
        format_id=format_id,
        upload_after=upload_after,
        upload_before=upload_before,
    )
    return {"total": total, "offset": offset, "limit": limit, "items": items}


@router.post("/snapshots")
def archive_snapshot(body: SnapshotRequest):
    return create_snapshot(body)


@router.get("/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str):
    return read_snapshot(snapshot_id)
