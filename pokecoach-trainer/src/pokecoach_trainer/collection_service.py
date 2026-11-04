"""Orchestration de collecte Showdown avec reprise et budgets."""

from __future__ import annotations

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
    JobKind,
    JobRecord,
    JobStatus,
)
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import write_json_atomic

SleepFn = Callable[[float], None]
ClockFn = Callable[[], float]


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
            f"max_new_replays={config.max_new_replays} "
            f"max_requests={config.max_requests} max_duration_s={config.max_duration_s}"
        )

    def budgets_exceeded() -> str | None:
        elapsed = clock_fn() - started
        if new_replays >= config.max_new_replays:
            return "max_new_replays"
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
            sleep_fn(config.request_interval_s)
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
                    sleep_fn(config.request_interval_s)
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


def run_collection_job(job_id: str) -> JobRecord:
    """Exécute un job de collecte dans le worker."""

    from pokecoach_trainer.jobs import release_training_lock

    record = get_job(job_id)
    if record.kind != JobKind.collection:
        raise InvalidConfiguration("job incompatible avec la collecte")
    record.status = JobStatus.running
    save_job(record)
    config = _config_from_job(record)
    job_log = CollectionJobLog(job_id)
    try:
        result = collect_replays(config, job_id, job_log=job_log)
        record.status = JobStatus.completed
        record.finished_at = datetime.now(timezone.utc)
        record.progress = result.model_dump(mode="json")
        record.result_ref = json.dumps(
            {"stop_reason": result.stop_reason, "new_replays": result.new_replays}
        )
        write_json_atomic(
            storage_path("jobs", job_id, "collection_result.json"),
            result.model_dump(mode="json"),
        )
        job_log.write(
            f"job completed new_replays={result.new_replays} stop_reason={result.stop_reason}"
        )
    except Exception as exc:
        record.status = JobStatus.failed
        record.error = str(exc)
        record.finished_at = datetime.now(timezone.utc)
        import traceback

        job_log.write(f"job failed: {exc}")
        job_log.write(traceback.format_exc())
        raise
    finally:
        release_training_lock(job_id)
    save_job(record)
    return record


def submit_collection(config: CollectionConfig) -> JobRecord:
    """Soumet un job de collecte via le mécanisme commun."""

    from pokecoach_trainer.job_service import submit_job

    payload = CollectionJobConfig(config=config)
    return submit_job(JobKind.collection, payload)
