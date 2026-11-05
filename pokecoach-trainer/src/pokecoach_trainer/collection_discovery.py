"""Découverte par fenêtres temporelles et couverture persistante."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import datetime, timezone

from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.collection_types import DiscoveryPage, TaskLease, WindowBounds
from pokecoach_trainer.errors import StaleLease
from pokecoach_trainer.schemas import CollectionConfig

_DAY_S = 86_400


def partition_range(
    source_format: str,
    start: int,
    end: int,
    *,
    partition_days: int,
) -> Iterator[WindowBounds]:
    """Découpe [start, end) sur une grille UTC ancrée à l'époque 0."""

    if start >= end:
        return
    step = partition_days * _DAY_S
    grid = (start // step) * step
    while grid < end:
        w_start = max(grid, start)
        w_end = min(grid + step, end)
        if w_start < w_end:
            yield WindowBounds(source_format=source_format, start=w_start, end=w_end)
        grid += step


def _window_id(bounds: WindowBounds, scan_generation: int) -> str:
    payload = f"{bounds.source_format}:{bounds.start}:{bounds.end}:{scan_generation}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _effective_bounds(config: CollectionConfig) -> tuple[int, int]:
    after = (
        int(config.upload_after.timestamp())
        if config.upload_after is not None
        else 0
    )
    before = config.effective_upload_before or config.upload_before
    if before is None:
        before = datetime.now(timezone.utc)
    end_ts = int(before.timestamp())
    return after, end_ts


def incremental_bounds(
    store: CollectionStore,
    source_format: str,
    *,
    end: int,
    overlap_hours: int,
) -> WindowBounds:
    """Fenêtre incrémentale [dernier repère - overlap, end)."""

    overlap_s = overlap_hours * 3600
    from pokecoach_trainer.archive_catalog import catalog_connection

    with catalog_connection(store._root) as conn:
        row = conn.execute(
            """
            SELECT MAX(end_ts) AS hi FROM coverage
            WHERE source_format=? AND proof_kind='scan_complete'
            """,
            (source_format,),
        ).fetchone()
        if row and row["hi"] is not None:
            start = max(0, int(row["hi"]) - overlap_s)
        else:
            start = max(0, end - overlap_s)
    return WindowBounds(source_format=source_format, start=start, end=end)


def seed_discovery(store: CollectionStore, campaign_id: str) -> int:
    """Matérialise jusqu'à 2× discovery_workers fenêtres non terminées."""

    view = store.get_campaign(campaign_id)
    config = view.effective_config
    scan_gen = 1 if config.reinspect else 0
    after, end_ts = _effective_bounds(config)
    if config.mode == "incremental":
        created = 0
        from pokecoach_trainer.formats import resolve_format

        for format_id in config.format_ids:
            support = resolve_format(format_id)
            for source_format in support.source_format_ids:
                bounds = incremental_bounds(
                    store, source_format, end=end_ts, overlap_hours=config.incremental_overlap_hours
                )
                created += store.ensure_window(campaign_id, bounds, scan_gen)
        return created

    from pokecoach_trainer.formats import resolve_format

    formats: list[str] = []
    for format_id in config.format_ids:
        formats.extend(resolve_format(format_id).source_format_ids)
    max_open = max(1, config.discovery_workers) * 2
    return store.materialize_windows(
        campaign_id,
        formats=formats,
        range_start=after,
        range_end=end_ts,
        partition_days=config.window_days,
        scan_generation=scan_gen,
        max_windows=max_open,
    )


def discovery_should_pause(store: CollectionStore, config: CollectionConfig) -> bool:
    """True si la file de téléchargement dépasse le seuil haut."""

    pending = store.pending_download_count()
    return pending >= config.queue_high_water


def process_search_page(
    store: CollectionStore,
    lease: TaskLease,
    page: DiscoveryPage,
) -> None:
    """Applique une page validée (IDs + curseur) dans une transaction."""

    if lease.window is None:
        raise StaleLease("lease découverte sans fenêtre")
    store.commit_discovery_page(lease, page)
