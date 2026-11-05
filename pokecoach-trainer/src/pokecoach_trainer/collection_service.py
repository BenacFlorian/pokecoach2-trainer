"""Orchestration de collecte Showdown avec reprise et budgets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pokecoach_trainer.archive_service import archive_replay, is_replay_archived
from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.errors import InvalidConfiguration
from pokecoach_trainer.formats import (
    is_champions_vgc_2026_replay_id,
    resolve_format,
    validate_champions_vgc_2026_showdown_id,
)
from pokecoach_trainer.jobs import get_job, save_job
from pokecoach_trainer.schemas import (
    ArchiveSource,
    CollectionConfig,
    CollectionJobConfig,
    CollectionResult,
    JobRecord,
)
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import write_json_atomic

SleepFn = Callable[[float], None]
ClockFn = Callable[[], float]


def normalize_collection_config(config: CollectionConfig, *, now: datetime) -> CollectionConfig:
    """Normalise formats/dates ; fige la borne haute technique au premier démarrage."""

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    formats = sorted(dict.fromkeys(config.format_ids))
    effective_before = config.upload_before
    if effective_before is None:
        effective_before = now
    return config.model_copy(
        update={
            "format_ids": formats,
            "effective_upload_before": effective_before,
        }
    )


def collection_request_fingerprint(config: CollectionConfig) -> str:
    """Empreinte stable de la requête utilisateur (sans borne technique figée)."""

    payload = config.model_dump(mode="json", exclude={"effective_upload_before"})
    if config.upload_before is None:
        payload["upload_before"] = None
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CollectionJobLog:
    """Journal texte flushé pour tail -f sur worker.log."""

    def __init__(self, job_id: str) -> None:
        self._path = storage_path("jobs", job_id, "worker.log")
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"{ts} {message}\n")
            handle.flush()


def _sync_job_progress(job_id: str, snapshot: dict[str, Any]) -> None:
    try:
        record = get_job(job_id)
        record.progress = snapshot
        save_job(record)
    except Exception:
        return


def _progress_path(job_id: str) -> Path:
    return storage_path("jobs", job_id, "collection_progress.json")


def _journal_path(job_id: str) -> Path:
    return storage_path("jobs", job_id, "collection_journal.jsonl")


def _load_progress(job_id: str) -> dict[str, Any]:
    path = _progress_path(job_id)
    if not path.is_file():
        return {"pending": [], "cursors": {}, "seen_pages": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_progress(job_id: str, progress: dict[str, Any]) -> None:
    write_json_atomic(_progress_path(job_id), progress)


def _append_journal(job_id: str, event: dict[str, Any]) -> None:
    path = _journal_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _config_from_job(record: JobRecord) -> CollectionConfig:
    cfg = record.config
    if isinstance(cfg, CollectionJobConfig):
        return cfg.config
    raise InvalidConfiguration("job sans configuration de collecte")


def _source_formats(config: CollectionConfig) -> list[tuple[str, str]]:
    """Associe format logique → identifiant Showdown source."""

    pairs: list[tuple[str, str]] = []
    for format_id in config.format_ids:
        support = resolve_format(format_id)
        if not support.collection_supported:
            raise InvalidConfiguration(
                f"collecte non supportée pour {format_id!r}",
                details={"format_id": format_id},
            )
        for source_id in support.source_format_ids:
            validate_champions_vgc_2026_showdown_id(source_id, field="source_format_ids")
            pairs.append((format_id, source_id))
    return pairs


def collect_replays(
    config: CollectionConfig,
    job_id: str,
    *,
    client: ShowdownReplayClient | None = None,
    sleep: SleepFn | None = None,
    clock: ClockFn | None = None,
    job_log: CollectionJobLog | None = None,
) -> CollectionResult:
    """
    Collecte incrémentale avec budgets, reprise et journalisation.

    Ne filtre aucun replay sur critères métier dataset.
    """

    import time

    sleep_fn = sleep or time.sleep
    clock_fn = clock or time.monotonic
    own_client = client is None
    http = client or ShowdownReplayClient(
        timeout_s=config.timeout_s,
        max_attempts=config.max_attempts,
        sleep=sleep_fn,
        clock=clock_fn,
    )
    started = clock_fn()
    progress = _load_progress(job_id)
    pending: list[str] = list(progress.get("pending") or [])
    cursors: dict[str, Any] = dict(progress.get("cursors") or {})
    seen_pages: set[str] = set(progress.get("seen_pages") or [])

    new_replays = 0
    skipped_existing = 0
    failed_replays = 0
    errors: list[dict[str, Any]] = []
    stop_reason = "completed"
    budget_limit: str | None = None
    coverage_incomplete = False

    source = ArchiveSource(
        provenance="showdown_public",
        fetched_at=datetime.now(timezone.utc),
    )

    if job_log:
        source_formats = [sid for _lid, sid in _source_formats(config)]
        job_log.write(
            "collecte démarrée "
            f"logical={config.format_ids} showdown={source_formats} "
            f"max_selected_replays={config.max_selected_replays} "
            f"max_requests={config.max_requests} max_duration_s={config.max_duration_s}"
        )

    def budgets_exceeded() -> str | None:
        elapsed = clock_fn() - started
        if new_replays >= config.max_selected_replays:
            return "max_selected_replays"
        if http.request_count >= config.max_requests:
            return "max_requests"
        if elapsed >= config.max_duration_s:
            return "max_duration_s"
        return None

    def persist_progress() -> None:
        _save_progress(
            job_id,
            {
                "pending": pending,
                "cursors": cursors,
                "seen_pages": sorted(seen_pages),
            },
        )
        _sync_job_progress(
            job_id,
            {
                "new_replays": new_replays,
                "skipped_existing": skipped_existing,
                "failed_replays": failed_replays,
                "pending": len(pending),
                "requests": http.request_count,
            },
        )

    def process_pending() -> bool:
        nonlocal stop_reason, budget_limit, new_replays, skipped_existing, failed_replays
        stopped = False
        while pending:
            limit = budgets_exceeded()
            if limit:
                stop_reason = "budget_reached"
                budget_limit = limit
                stopped = True
                break
            replay_id = pending.pop(0)
            if not is_champions_vgc_2026_replay_id(replay_id):
                failed_replays += 1
                errors.append(
                    {"replay_id": replay_id, "error": "replay hors périmètre Champions VGC 2026"}
                )
                if job_log:
                    job_log.write(f"rejeté {replay_id} (hors gen9championsvgc2026)")
                _append_journal(job_id, {"event": "rejected", "replay_id": replay_id})
                persist_progress()
                continue
            if is_replay_archived(replay_id):
                skipped_existing += 1
                if job_log and skipped_existing % 50 == 0:
                    job_log.write(
                        f"skipped_existing={skipped_existing} (dernier {replay_id})"
                    )
                persist_progress()
                continue
            sleep_fn(config.legacy_request_interval_s)
            try:
                payload = http.fetch(replay_id)
            except InvalidConfiguration as exc:
                failed_replays += 1
                errors.append({"replay_id": replay_id, "error": str(exc)})
                _append_journal(job_id, {"event": "fetch_failed", "replay_id": replay_id})
                if job_log:
                    job_log.write(f"fetch_failed {replay_id}: {exc}")
                persist_progress()
                continue
            uploadtime = payload.get("uploadtime")
            if uploadtime is not None:
                ts = datetime.fromtimestamp(int(uploadtime), tz=timezone.utc)
                if config.upload_after and ts < config.upload_after.replace(tzinfo=timezone.utc):
                    persist_progress()
                    continue
                if config.upload_before and ts > config.upload_before.replace(tzinfo=timezone.utc):
                    persist_progress()
                    continue
            archive_replay(replay_id, payload, source)
            new_replays += 1
            _append_journal(job_id, {"event": "archived", "replay_id": replay_id})
            if job_log:
                job_log.write(
                    f"archived {replay_id} "
                    f"(new={new_replays} requests={http.request_count} pending={len(pending)})"
                )
            persist_progress()
            limit = budgets_exceeded()
            if limit:
                stop_reason = "budget_reached"
                budget_limit = limit
                stopped = True
                break
        return stopped

    try:
        if process_pending():
            pass
        else:
            for _logical_id, source_format in _source_formats(config):
                cursor = cursors.get(source_format)
                while True:
                    limit = budgets_exceeded()
                    if limit:
                        stop_reason = "budget_reached"
                        budget_limit = budget_limit or limit
                        break
                    sleep_fn(config.legacy_request_interval_s)
                    page = http.search(source_format, cursor)
                    if not page.replay_ids and page.next_cursor is None:
                        break
                    signature = json.dumps(page.next_cursor or {}, sort_keys=True)
                    if signature in seen_pages:
                        stop_reason = "coverage_incomplete"
                        coverage_incomplete = True
                        break
                    seen_pages.add(signature)
                    allowed = [
                        rid
                        for rid in page.replay_ids
                        if rid not in pending and is_champions_vgc_2026_replay_id(rid)
                    ]
                    rejected_n = len(page.replay_ids) - len(
                        [rid for rid in page.replay_ids if is_champions_vgc_2026_replay_id(rid)]
                    )
                    if rejected_n and job_log:
                        job_log.write(
                            f"search {source_format}: {rejected_n} replays ignorés "
                            "(hors gen9championsvgc2026)"
                        )
                    discovered = allowed
                    pending.extend(discovered)
                    _append_journal(
                        job_id,
                        {
                            "event": "discovered",
                            "format": source_format,
                            "count": len(discovered),
                        },
                    )
                    if job_log:
                        job_log.write(
                            f"search {source_format} discovered={len(discovered)} "
                            f"pending={len(pending)} requests={http.request_count} "
                            f"cursor={cursor!r}"
                        )
                    cursor = page.next_cursor
                    cursors[source_format] = cursor
                    persist_progress()
                    if process_pending():
                        break
                    if cursor is None:
                        break
                if stop_reason != "completed":
                    break
            if stop_reason == "completed":
                process_pending()
    finally:
        if own_client:
            http.close()

    if coverage_incomplete and stop_reason == "completed":
        stop_reason = "coverage_incomplete"

    if job_log:
        job_log.write(
            "collecte terminée "
            f"stop_reason={stop_reason} new_replays={new_replays} "
            f"skipped_existing={skipped_existing} failed_replays={failed_replays} "
            f"requests={http.request_count} budget_limit={budget_limit!r}"
        )

    result = CollectionResult(
        new_replays=new_replays,
        skipped_existing=skipped_existing,
        failed_replays=failed_replays,
        total_requests=http.request_count,
        stop_reason=stop_reason,
        budget_limit=budget_limit,
        coverage_incomplete=coverage_incomplete,
        errors=errors,
        effective_config=config,
    )
    return result


def submit_collection(config: CollectionConfig) -> JobRecord:
    """Démarre une campagne parallèle supervisée."""

    from pokecoach_trainer.collection_store import CollectionStore
    from pokecoach_trainer.collection_supervisor import ensure_supervisor
    from pokecoach_trainer.settings import get_settings

    root = get_settings().storage_root
    store = CollectionStore(root)
    view = store.start_campaign(config)
    ensure_supervisor(root, view.job.job_id)
    return view.job


def get_collection(job_id: str):
    from pokecoach_trainer.collection_store import CollectionStore
    from pokecoach_trainer.errors import NotFound
    from pokecoach_trainer.jobs import get_job
    from pokecoach_trainer.schemas import JobKind
    from pokecoach_trainer.settings import get_settings

    store = CollectionStore(get_settings().storage_root)
    try:
        return store.get_campaign(job_id)
    except NotFound:
        try:
            record = get_job(job_id)
        except FileNotFoundError as exc:
            raise NotFound(f"job {job_id} introuvable") from exc
        if record.kind != JobKind.collection:
            raise NotFound(f"campagne {job_id} introuvable")
        raise NotFound(
            "ancien job de collecte JSON ; sources éventuellement supprimées — "
            "lancer une nouvelle collecte (non reprenable)",
            details={"job_id": job_id, "legacy": True},
        )


def pause_collection(job_id: str):
    from pokecoach_trainer.collection_store import CollectionStore
    from pokecoach_trainer.settings import get_settings

    store = CollectionStore(get_settings().storage_root)
    return store.command(job_id, "pause")


def stop_collection(job_id: str):
    from pokecoach_trainer.collection_store import CollectionStore
    from pokecoach_trainer.settings import get_settings

    store = CollectionStore(get_settings().storage_root)
    return store.command(job_id, "stop")


def resume_collection(job_id: str, request):
    from pokecoach_trainer.archive_catalog import catalog_connection
    from pokecoach_trainer.collection_store import CollectionStore
    from pokecoach_trainer.collection_supervisor import ensure_supervisor
    from pokecoach_trainer.errors import InvalidConfiguration, NotFound
    from pokecoach_trainer.jobs import get_job
    from pokecoach_trainer.schemas import JobKind
    from pokecoach_trainer.settings import get_settings

    root = get_settings().storage_root
    store = CollectionStore(root)
    with catalog_connection(root) as conn:
        row = conn.execute(
            "SELECT 1 FROM campaigns WHERE campaign_id = ?", (job_id,)
        ).fetchone()
    if row is None:
        try:
            record = get_job(job_id)
        except FileNotFoundError as exc:
            raise NotFound(f"job {job_id} introuvable") from exc
        if record.kind == JobKind.collection:
            raise InvalidConfiguration(
                "ancien job non reprenable, lancer une nouvelle collecte",
                details={"job_id": job_id},
            )
        raise NotFound(f"campagne {job_id} introuvable")
    view = store.command(
        job_id, "resume", new_budgets=getattr(request, "budgets", None)
    )
    ensure_supervisor(root, job_id)
    return view


def scale_collection(job_id: str, request):
    from pokecoach_trainer.collection_store import CollectionStore
    from pokecoach_trainer.settings import get_settings

    store = CollectionStore(get_settings().storage_root)
    return store.scale(
        job_id,
        discovery_workers=request.discovery_workers,
        download_workers=request.download_workers,
    )


def retry_failed_collection(job_id: str, request):
    from pokecoach_trainer.collection_store import CollectionStore
    from pokecoach_trainer.settings import get_settings

    store = CollectionStore(get_settings().storage_root)
    return store.retry_failed(job_id, request)


def run_collection_job(job_id: str) -> JobRecord:
    """Compatibilité worker : redirige vers le superviseur."""

    from pokecoach_trainer.collection_supervisor import ensure_supervisor
    from pokecoach_trainer.settings import get_settings

    ensure_supervisor(get_settings().storage_root, job_id)
    return get_job(job_id)
