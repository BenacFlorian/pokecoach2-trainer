from __future__ import annotations

import threading
from datetime import datetime, timezone

from pokecoach_trainer.archive_catalog import catalog_connection
from pokecoach_trainer.archive_service import archive_replay
from pokecoach_trainer.collection_filters import evaluate_collection_candidate
from pokecoach_trainer.collection_publication import (
    build_replay_envelope,
    prepare_replay,
    publish_replay,
)
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem
from pokecoach_trainer.schemas import ArchiveSource, CollectionConfig, ReplayFilterConfig
from pokecoach_trainer.settings import clear_settings_cache
from replay_log_fixtures import build_log

SOURCE = "gen9championsvgc2026regmc"


class FakeClock(SystemClock):
    def __init__(self) -> None:
        self._wall = 1_000.0
        self._mono = 100.0
        self._boot = "boot-a"

    def wall(self) -> float:
        return self._wall

    def monotonic(self) -> float:
        return self._mono

    def boot_id(self) -> str:
        return self._boot


def _cfg(**kwargs):
    base = {
        "format_ids": ["mc"],
        "max_selected_replays": 5,
        "requests_per_s": 20.0,
        "dataset_filters": ReplayFilterConfig(),
    }
    base.update(kwargs)
    return CollectionConfig(**base)


def _payload(replay_id: str, log: str | None = None) -> dict:
    return {
        "id": replay_id,
        "uploadtime": 1,
        "log": log or build_log(p1_rating=1300, p2_rating=900),
        "format": SOURCE,
    }


def test_rejected_replay_releases_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(_cfg())
    rid = f"{SOURCE}-rej"
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    ref, env = build_replay_envelope(rid, _payload(rid, build_log(include_ots=False)), source)
    evaluation = evaluate_collection_candidate(view.effective_config, ref, env)
    before = store.get_campaign(view.job.job_id).counters
    outcome = store.record_evaluation(claim.lease, ref, evaluation)
    assert outcome == "rejected"
    after = store.get_campaign(view.job.job_id).counters
    assert after["selected_in_attempt"] == before["selected_in_attempt"]
    assert after["reserved_slots"] == 0


def test_selection_counts_only_after_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(_cfg())
    rid = f"{SOURCE}-ok"
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    payload = _payload(rid)
    ref, env = build_replay_envelope(rid, payload, source)
    evaluation = evaluate_collection_candidate(view.effective_config, ref, env)
    assert evaluation.selection.selected
    outcome = store.record_evaluation(claim.lease, ref, evaluation)
    assert outcome == "ready"
    mid = store.get_campaign(view.job.job_id).counters
    assert mid["selected_in_attempt"] == 0
    prepared = prepare_replay(tmp_path, rid, payload, source)
    result = publish_replay(store, prepared, lease=claim.lease)
    first = store.finalize_selection(claim.lease, result.ref, new_version=result.new_version)
    assert first is True
    after = store.get_campaign(view.job.job_id).counters
    assert after["selected_in_attempt"] == 1


def test_finalize_selection_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(_cfg())
    rid = f"{SOURCE}-idemp"
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    payload = _payload(rid)
    archive_replay(rid, payload, source)
    ref, env = build_replay_envelope(rid, payload, source)
    evaluation = evaluate_collection_candidate(view.effective_config, ref, env)
    store.record_evaluation(claim.lease, ref, evaluation)
    assert store.finalize_selection(claim.lease, ref, new_version=False) is True
    # lease already cleared; recover path must not increment again
    again = store.finalize_recovered_publication(
        campaign_id=view.job.job_id,
        attempt_id=view.attempt_id,
        ref=ref,
        new_version=False,
    )
    assert again is False
    assert store.get_campaign(view.job.job_id).counters["selected_in_attempt"] == 1


def test_rejected_alias_does_not_reserve_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(_cfg())
    rid = f"{SOURCE}-alias"
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    ref, env = build_replay_envelope(rid, _payload(rid, build_log(include_ots=False)), source)
    evaluation = evaluate_collection_candidate(view.effective_config, ref, env)
    store.record_evaluation(claim.lease, ref, evaluation)
    with store._tx() as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM campaign_log_claims").fetchone()["n"]
    assert n == 0


def _admit(store, lease, replay_id, payload):
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    ref, env = build_replay_envelope(replay_id, payload, source)
    evaluation = evaluate_collection_candidate(
        store.get_campaign(lease.campaign_id).effective_config, ref, env
    )
    return ref, env, store.record_evaluation(lease, ref, evaluation)


def test_two_workers_quota_one_selects_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(_cfg(max_selected_replays=1))
    ids = [f"{SOURCE}-q{i}" for i in range(2)]
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE) for rid in ids])
    barrier = threading.Barrier(2)
    leases: list = []

    def worker(wid: str) -> None:
        local = CollectionStore(tmp_path, clock=clock)
        barrier.wait()
        claim = local.claim_download(view.job.job_id, wid)
        leases.append(claim.lease)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    held = [lease for lease in leases if lease is not None]
    assert len(held) == 1
    lease = held[0]
    payload = _payload(lease.replay_id)
    ref, _env, outcome = _admit(store, lease, lease.replay_id, payload)
    assert outcome == "ready"
    archive_replay(
        lease.replay_id,
        payload,
        ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc)),
    )
    assert store.finalize_selection(lease, ref, new_version=True) is True
    after = store.get_campaign(view.job.job_id).counters
    assert after["selected_in_attempt"] == 1
    with catalog_connection(tmp_path) as conn:
        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_log_claims WHERE state='pending'"
        ).fetchone()["n"]
    assert pending == 0


def test_concurrent_equal_logs_select_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(_cfg())
    log = build_log(p1_rating=1300, p2_rating=900)
    ids = [f"{SOURCE}-dup{i}" for i in range(2)]
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE) for rid in ids])
    claims = [store.claim_download(view.job.job_id, f"w{i}") for i in range(2)]
    assert all(claim.lease for claim in claims)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def worker(lease, replay_id: str) -> None:
        local = CollectionStore(tmp_path, clock=clock)
        payload = _payload(replay_id, log)
        barrier.wait()
        _ref, _env, outcome = _admit(local, lease, replay_id, payload)
        outcomes.append(outcome)

    threads = [
        threading.Thread(target=worker, args=(claim.lease, claim.lease.replay_id))
        for claim in claims
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("ready") == 1
    assert outcomes.count("wait") + outcomes.count("duplicate") == 1


def test_pending_duplicate_retries_if_owner_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(_cfg())
    log = build_log(p1_name="Same", p1_rating=1300, p2_rating=900)
    owner_id = f"{SOURCE}-own"
    waiter_id = f"{SOURCE}-wait"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(owner_id, 1, SOURCE), SearchItem(waiter_id, 1, SOURCE)],
    )
    owner = store.claim_download(view.job.job_id, "owner")
    waiter = store.claim_download(view.job.job_id, "waiter")
    assert owner.lease and waiter.lease
    _ref, _env, first = _admit(store, owner.lease, owner_id, _payload(owner_id, log))
    _ref2, _env2, second = _admit(store, waiter.lease, waiter_id, _payload(waiter_id, log))
    assert first == "ready"
    assert second == "wait"
    store.fail_lease(owner.lease, permanent=True, unavailable=False, error="owner-fail")
    clock._wall += 3.0
    retry = store.claim_download(view.job.job_id, "waiter2")
    assert retry.lease is not None
    assert retry.lease.replay_id == waiter_id
    _ref3, _env3, again = _admit(store, retry.lease, waiter_id, _payload(waiter_id, log))
    assert again == "ready"


def test_deduplicate_false_keeps_distinct_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(_cfg(dataset_filters=ReplayFilterConfig(deduplicate=False)))
    log = build_log(p1_name="Twin", p1_rating=1300, p2_rating=900)
    ids = [f"{SOURCE}-keep{i}" for i in range(2)]
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE) for rid in ids])
    outcomes = []
    for i, rid in enumerate(ids):
        claim = store.claim_download(view.job.job_id, f"w{i}")
        assert claim.lease
        payload = _payload(rid, log)
        ref, _env, outcome = _admit(store, claim.lease, rid, payload)
        assert outcome == "ready"
        archive_replay(
            rid,
            payload,
            ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc)),
        )
        assert store.finalize_selection(claim.lease, ref, new_version=True) is True
        outcomes.append(outcome)
    after = store.get_campaign(view.job.job_id).counters
    assert after["selected_in_attempt"] == 2
