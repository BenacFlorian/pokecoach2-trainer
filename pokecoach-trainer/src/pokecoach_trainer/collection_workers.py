"""Boucles workers collecte (découverte et téléchargement)."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.collection_discovery import (
    discovery_should_pause,
    process_search_page,
    seed_discovery,
)
from pokecoach_trainer.collection_filters import (
    evaluate_collection_candidate,
    evaluate_selection,
    extract_team_summary,
)
from pokecoach_trainer.collection_limits import acquire_request_permit, record_request_result
from pokecoach_trainer.collection_logging import log_worker
from pokecoach_trainer.collection_publication import (
    build_replay_envelope,
    prepare_replay,
    publish_replay,
    recover_publications,
)
from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.collection_types import Clock, CollectionEvaluation, SelectionDecision, TaskLease
from pokecoach_trainer.errors import StaleLease
from pokecoach_trainer.schemas import ArchiveRef, ArchiveSource, CollectionConfig

SleepFn = Callable[[float], None]
ClientFactory = Callable[[CollectionConfig], ShowdownReplayClient]


def _interruptible_sleep(stop: threading.Event, seconds: float, sleep: SleepFn) -> bool:
    if seconds <= 0:
        return stop.is_set()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop.is_set():
            return True
        sleep(min(0.25, deadline - time.monotonic()))
    return stop.is_set()


def _heartbeat_loop(
    store: CollectionStore,
    lease_holder: list[TaskLease | None],
    stop: threading.Event,
    config: CollectionConfig,
    sleep: SleepFn,
) -> None:
    while not stop.is_set():
        lease = lease_holder[0]
        if lease is not None:
            try:
                lease_holder[0] = store.renew_lease(lease)
            except StaleLease:
                stop.set()
                return
        _interruptible_sleep(stop, config.heartbeat_s, sleep)


def run_collection_worker(
    root: Path,
    campaign_id: str,
    worker_id: str,
    role: Literal["discovery", "download"],
    *,
    client_factory: ClientFactory | None = None,
    stop_event: threading.Event | None = None,
    sleep: SleepFn | None = None,
    max_steps: int | None = None,
    clock: Clock | None = None,
) -> int:
    """Exécute une boucle worker jusqu'à arrêt ou fin de campagne."""

    sleep_fn = sleep or time.sleep
    stop = stop_event or threading.Event()
    store = CollectionStore(root, clock=clock) if clock is not None else CollectionStore(root)
    view = store.get_campaign(campaign_id)
    config = view.effective_config
    recover_publications(store)
    if config.dataset_filters is not None:
        from pokecoach_trainer.archive_catalog import catalog_connection

        with catalog_connection(root) as conn:
            store._assert_validation_versions(conn, campaign_id, config)
    client = (client_factory or _default_client_factory)(config)
    lease_holder: list[TaskLease | None] = [None]
    hb_stop = threading.Event()
    hb = threading.Thread(
        target=_heartbeat_loop,
        args=(store, lease_holder, hb_stop, config, sleep_fn),
        daemon=True,
    )
    hb.start()
    steps = 0
    log_worker(campaign_id, worker_id, f"worker {role} démarré")
    try:
        if role == "discovery":
            seed_discovery(store, campaign_id)
        while not stop.is_set():
            if max_steps is not None and steps >= max_steps:
                break
            view = store.get_campaign(campaign_id)
            if view.phase not in {"running", "queued", "pausing", "stopping"}:
                break
            if role == "discovery":
                if discovery_should_pause(store, config):
                    if _interruptible_sleep(stop, 0.5, sleep_fn):
                        break
                    continue
                claim = store.claim_window(campaign_id, worker_id)
                if claim.lease is None and claim.reason == "no_windows":
                    seed_discovery(store, campaign_id)
                    if _interruptible_sleep(stop, 0.2, sleep_fn):
                        break
                    continue
            else:
                claim = store.claim_download(campaign_id, worker_id)
            if claim.lease is None:
                if max_steps is not None and claim.reason in {
                    "queue_empty",
                    "selection_full",
                    "campaign_not_running",
                }:
                    break
                if _interruptible_sleep(stop, claim.wait_s, sleep_fn):
                    break
                continue
            lease_holder[0] = claim.lease
            steps += 1
            try:
                if role == "discovery":
                    _run_discovery_step(store, claim.lease, config, client, stop, sleep_fn)
                else:
                    _run_download_step(store, claim.lease, config, client, stop, sleep_fn)
            except StaleLease:
                break
            finally:
                lease_holder[0] = None
    finally:
        hb_stop.set()
        hb.join(timeout=5)
        client.close()
    return steps


def _default_client_factory(config: CollectionConfig) -> ShowdownReplayClient:
    return ShowdownReplayClient(timeout_s=config.timeout_s, max_attempts=config.max_attempts)


def _run_discovery_step(
    store: CollectionStore,
    lease: TaskLease,
    config: CollectionConfig,
    client: ShowdownReplayClient,
    stop: threading.Event,
    sleep: SleepFn,
) -> None:
    if lease.window is None or lease.cursor_before is None:
        return
    permit = acquire_request_permit(store, lease, "search")
    while permit.permit is None and permit.wait_s > 0:
        if _interruptible_sleep(stop, permit.wait_s, sleep):
            return
        permit = acquire_request_permit(store, lease, "search")
    if permit.permit is None:
        return
    try:
        page = client.search_once(lease.window.source_format, before=lease.cursor_before)
        record_request_result(store, permit.permit, status_code=200, elapsed_s=0.0)
    except Exception:
        record_request_result(store, permit.permit, status_code=500, elapsed_s=0.0)
        store.release_lease(lease)
        return
    process_search_page(store, lease, page)
    n = len(page.items)
    log_worker(
        lease.campaign_id,
        lease.worker_id,
        f"search before={lease.cursor_before} items={n} exhausted={page.exhausted}",
    )


def _admit_payload(
    store: CollectionStore,
    lease: TaskLease,
    config: CollectionConfig,
    payload: dict,
    *,
    local: bool,
) -> None:
    from pokecoach_trainer.errors import InvalidConfiguration

    replay_id = lease.replay_id
    assert replay_id
    source = ArchiveSource(
        provenance="archive_reuse" if local else "showdown_public",
        fetched_at=datetime.now(timezone.utc),
    )
    try:
        ref, envelope = build_replay_envelope(replay_id, payload, source)
    except InvalidConfiguration as exc:
        decision = SelectionDecision(
            selected=False,
            reason="invalid_replay_payload",
            filter_sha256=evaluate_selection(config, uploadtime=None, summary=None).filter_sha256,
        )
        store.record_evaluation(
            lease,
            ArchiveRef(
                replay_id=replay_id,
                content_sha256="",
                log_sha256="",
                format_id=str(payload.get("format") or ""),
                object_relpath=f"objects/{replay_id}/invalid.json",
            ),
            CollectionEvaluation(
                selection=decision, metadata=None, eligibility=None, summary=None
            ),
        )
        log_worker(
            lease.campaign_id,
            lease.worker_id,
            f"replay {replay_id} rejeté (invalid_replay_payload) {exc}",
        )
        return
    evaluation = evaluate_collection_candidate(config, ref, envelope)
    outcome = store.record_evaluation(lease, ref, evaluation)
    if outcome == "rejected":
        log_worker(
            lease.campaign_id,
            lease.worker_id,
            f"replay {replay_id} rejeté ({evaluation.selection.reason})",
        )
        return
    if outcome == "duplicate":
        log_worker(
            lease.campaign_id,
            lease.worker_id,
            f"replay {replay_id} doublon log",
        )
        return
    if outcome == "wait":
        log_worker(
            lease.campaign_id,
            lease.worker_id,
            f"replay {replay_id} wait hash concurrent",
        )
        return
    if local:
        store.finalize_selection(lease, ref, new_version=False)
        log_worker(
            lease.campaign_id,
            lease.worker_id,
            f"replay {replay_id} réutilisé localement",
        )
        return
    prepared = prepare_replay(store._root, replay_id, payload, source)
    result = publish_replay(store, prepared, lease=lease)
    store.finalize_selection(lease, result.ref, new_version=result.new_version)
    log_worker(
        lease.campaign_id,
        lease.worker_id,
        f"replay {replay_id} publié new_version={result.new_version}",
    )


def _run_download_step(
    store: CollectionStore,
    lease: TaskLease,
    config: CollectionConfig,
    client: ShowdownReplayClient,
    stop: threading.Event,
    sleep: SleepFn,
) -> None:
    replay_id = lease.replay_id
    if not replay_id:
        return
    from pokecoach_trainer.archive_catalog import catalog_has_replay, catalog_versions
    from pokecoach_trainer.archive_service import read_archived_object
    from pokecoach_trainer.errors import InvalidConfiguration, NotFound

    if catalog_has_replay(store._root, replay_id):
        if config.dataset_filters is None:
            decision = evaluate_selection(config, uploadtime=None, summary=None)
            store.record_selection(lease, decision, None)
            log_worker(
                lease.campaign_id,
                lease.worker_id,
                f"replay {replay_id} déjà archivé decision={decision.reason}",
            )
            if decision.selected:
                with store._tx() as conn:
                    conn.execute(
                        """
                        UPDATE download_tasks SET state='archived', reserved_slot=0
                        WHERE replay_id=? AND lease_token=?
                        """,
                        (replay_id, lease.token),
                    )
            else:
                store.release_lease(lease)
            return
        versions = catalog_versions(store._root, replay_id)
        latest = versions[-1] if versions else None
        if latest is not None:
            try:
                envelope = read_archived_object(replay_id, latest.content_sha256)
                payload = envelope.get("provider") or envelope
                if payload.get("id") != replay_id:
                    payload = {**payload, "id": replay_id}
                _admit_payload(store, lease, config, payload, local=True)
                return
            except (NotFound, InvalidConfiguration, ValueError, OSError, KeyError):
                pass
    permit = acquire_request_permit(store, lease, "fetch")
    while permit.permit is None and permit.wait_s > 0:
        if _interruptible_sleep(stop, permit.wait_s, sleep):
            return
        permit = acquire_request_permit(store, lease, "fetch")
    if permit.permit is None:
        return
    try:
        payload = client.fetch_once(replay_id)
        record_request_result(store, permit.permit, status_code=200, elapsed_s=0.0)
    except Exception as exc:
        record_request_result(store, permit.permit, status_code=500, elapsed_s=0.0)
        store.fail_lease(lease, permanent=False, unavailable=False, error=str(exc))
        return
    if config.dataset_filters is not None:
        try:
            _admit_payload(store, lease, config, payload, local=False)
        except Exception:
            log_worker(
                lease.campaign_id,
                lease.worker_id,
                f"replay {replay_id} erreur interne d'admission",
            )
            store.fail_lease(lease, permanent=False, unavailable=False, error="admission_internal")
        return
    summary = extract_team_summary(payload)
    uploadtime = payload.get("uploadtime")
    ts = int(uploadtime) if uploadtime is not None else None
    decision = evaluate_selection(config, uploadtime=ts, summary=summary)
    store.record_selection(lease, decision, summary)
    if not decision.selected:
        log_worker(
            lease.campaign_id,
            lease.worker_id,
            f"replay {replay_id} rejeté ({decision.reason})",
        )
        return
    source = ArchiveSource(provenance="showdown_public", fetched_at=datetime.now(timezone.utc))
    prepared = prepare_replay(store._root, replay_id, payload, source)
    result = publish_replay(store, prepared, lease=lease)
    log_worker(
        lease.campaign_id,
        lease.worker_id,
        f"replay {replay_id} publié new_version={result.new_version}",
    )
