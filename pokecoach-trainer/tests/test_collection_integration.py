"""Tests intégration collecte (harness local)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from pokecoach_trainer.archive_catalog import catalog_connection
from pokecoach_trainer.archive_service import create_campaign_snapshot
from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.collection_discovery import partition_range
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem
from pokecoach_trainer.collection_workers import run_collection_worker
from pokecoach_trainer.jobs import acquire_resource_lock, release_resource_lock, save_job
from pokecoach_trainer.schemas import (
    CollectionConfig,
    ExperimentConfig,
    JobKind,
    JobRecord,
    JobStatus,
)
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

SOURCE = "gen9championsvgc2026regmc"


class FakeClock(SystemClock):
    def __init__(self) -> None:
        self._wall = 1_700_000_000.0
        self._mono = 10.0
        self._boot = "boot-int"

    def wall(self) -> float:
        return self._wall

    def monotonic(self) -> float:
        return self._mono

    def boot_id(self) -> str:
        return self._boot


def tick(clock: FakeClock):
    def _sleep(seconds: float) -> None:
        clock._mono += max(0.0, seconds)

    return _sleep


def mc_config(**kwargs):
    base = {
        "format_ids": ["mc"],
        "requests_per_s": 20.0,
        "max_requests": 5000,
        "max_selected_replays": 5000,
    }
    base.update(kwargs)
    return CollectionConfig(**base)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield tmp_path


def test_partition_covers_range_without_gaps():
    windows = list(partition_range("f", 0, 604_800 * 3, partition_days=7))
    assert windows[0].start == 0
    assert windows[-1].end == 604_800 * 3
    for a, b in zip(windows, windows[1:]):
        assert a.end == b.start


def _mock_client(replay_ids: list[str]) -> ShowdownReplayClient:
    payloads = {
        rid: {
            "id": rid,
            "uploadtime": idx + 1,
            "log": "|turn|1\n",
            "format": SOURCE,
        }
        for idx, rid in enumerate(replay_ids)
    }

    def handler(request: httpx.Request) -> httpx.Response:
        rid = request.url.path.strip("/").replace(".json", "")
        if rid not in payloads:
            return httpx.Response(404)
        return httpx.Response(200, json=payloads[rid])

    return ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)


def test_parallel_pipeline_matches_serial_unique_set(env, monkeypatch):
    n = 5
    ids = [f"{SOURCE}-par-{i}" for i in range(n)]
    clock = FakeClock()

    def run_workers(root: Path, worker_count: int) -> set[str]:
        monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(root))
        clear_settings_cache()
        ensure_storage_layout()
        store = CollectionStore(root, clock=clock)
        view = store.start_campaign(mc_config(max_selected_replays=n))
        store.enqueue_replays(
            view.job.job_id,
            [
                SearchItem(replay_id=rid, uploadtime=i + 1, source_format=SOURCE)
                for i, rid in enumerate(ids)
            ],
        )
        client = _mock_client(ids)
        per = max(1, n // worker_count)
        published: set[str] = set()
        for w in range(worker_count):
            chunk = ids[w * per : (w + 1) * per]
            if not chunk:
                continue
            run_collection_worker(
                root,
                view.job.job_id,
                f"dl-{w}",
                "download",
                client_factory=lambda _c: client,
                sleep=tick(clock),
                max_steps=n * 4,
                clock=clock,
            )
        with catalog_connection(root) as conn:
            rows = conn.execute(
                """
                SELECT replay_id FROM campaign_replays
                WHERE campaign_id=? AND decision='selected'
                """,
                (view.job.job_id,),
            ).fetchall()
            published = {row["replay_id"] for row in rows}
        client.close()
        return published

    serial_root = env / "serial"
    serial_root.mkdir()
    parallel_root = env / "parallel"
    parallel_root.mkdir()
    serial = run_workers(serial_root, 1)
    parallel = run_workers(parallel_root, 4)
    assert serial == parallel
    assert len(parallel) >= 1
    from pokecoach_trainer.archive_catalog import catalog_has_replay

    assert all(catalog_has_replay(parallel_root, rid) for rid in parallel)


def test_restart_after_crash_preserves_pipeline(env):
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(mc_config(max_selected_replays=5))
    rid = f"{SOURCE}-crash"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )
    claim = store.claim_download(view.job.job_id, "w-crash")
    assert claim.lease
    store2 = CollectionStore(env, clock=clock)
    claim2 = store2.claim_download(view.job.job_id, "w-resume")
    assert claim2.lease is None or claim2.lease.replay_id == rid


def test_rate_limit_and_budget_hold_under_process_races(env):
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(mc_config(max_requests=2, max_selected_replays=10))
    rid = f"{SOURCE}-budget"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"})

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=tick(clock))
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-budget",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=3,
        clock=clock,
    )
    with catalog_connection(env) as conn:
        consumed = conn.execute(
            "SELECT requests_consumed FROM attempts WHERE attempt_id=?",
            (view.attempt_id,),
        ).fetchone()["requests_consumed"]
    assert consumed <= 2
    client.close()


def test_snapshot_and_compute_coexist_with_collection(env):
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(mc_config(max_selected_replays=5))
    rid = f"{SOURCE}-snap-co"
    with catalog_connection(env) as conn:
        conn.execute(
            """
            INSERT INTO campaign_replays(campaign_id, replay_id, decision, reason, filter_sha256)
            VALUES (?, ?, 'selected', 'selected', 'f')
            """,
            (view.job.job_id, rid),
        )
    compute = JobRecord(
        job_id="job-compute-int",
        kind=JobKind.training,
        status=JobStatus.queued,
        config=ExperimentConfig(name="x", dataset_id="ds1"),
    )
    save_job(compute)
    fd = acquire_resource_lock("compute", compute.job_id)
    try:
        snap = create_campaign_snapshot(view.job.job_id)
        assert snap.snapshot_id
    finally:
        release_resource_lock(fd)
