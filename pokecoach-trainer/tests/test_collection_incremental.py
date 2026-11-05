"""Tests mode incrémental."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from pokecoach_trainer.archive_catalog import catalog_connection
from pokecoach_trainer.collection_discovery import incremental_bounds, seed_discovery
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem
from pokecoach_trainer.collection_workers import run_collection_worker
from pokecoach_trainer.schemas import CollectionConfig
from pokecoach_trainer.settings import clear_settings_cache


class FakeClock(SystemClock):
    def __init__(self) -> None:
        self._wall = 2_000_000.0
        self._mono = 1.0
        self._boot = "b"

    def wall(self) -> float:
        return self._wall

    def monotonic(self) -> float:
        return self._mono

    def boot_id(self) -> str:
        return self._boot


SOURCE = "gen9championsvgc2026regmc"


def tick(clock: FakeClock):
    def _sleep(seconds: float) -> None:
        clock._mono += max(0.0, seconds)

    return _sleep


def mc_config(**kwargs):
    base = {"format_ids": ["mc"], "requests_per_s": 20.0, "max_requests": 500}
    base.update(kwargs)
    return CollectionConfig(**base)


def test_incremental_without_watermark_covers_last_48h(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path, clock=FakeClock())
    end = 1_000_000
    bounds = incremental_bounds(store, "fmt", end=end, overlap_hours=48)
    assert bounds.end == end
    assert bounds.start == end - 48 * 3600


def test_incremental_unchanged_downloads_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(mc_config(mode="incremental", max_selected_replays=5))
    rid = f"{SOURCE}-same"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=int(clock.wall()), source_format=SOURCE)],
    )
    with catalog_connection(tmp_path) as conn:
        conn.execute(
            """
            INSERT INTO campaign_replays(campaign_id, replay_id, decision, reason, filter_sha256)
            VALUES (?, ?, 'selected', 'selected', 'x')
            """,
            (view.job.job_id, rid),
        )
        conn.execute("UPDATE download_tasks SET state='archived' WHERE replay_id=?", (rid,))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("aucun GET attendu")

    from pokecoach_trainer.collection_client import ShowdownReplayClient

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        tmp_path,
        view.job.job_id,
        "dl-inc",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    client.close()


def test_incremental_ten_new_ids_downloads_ten(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    view = store.start_campaign(mc_config(mode="incremental", max_selected_replays=20))
    items = [
        SearchItem(replay_id=f"{SOURCE}-inc{i}", uploadtime=1_000 + i, source_format=SOURCE)
        for i in range(10)
    ]
    store.enqueue_replays(view.job.job_id, items)
    fetched = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fetched
        if request.url.path.endswith(".json"):
            fetched += 1
            rid = request.url.path.split("/")[-1].replace(".json", "")
            return httpx.Response(
                200,
                json={"id": rid, "uploadtime": 1, "log": "|turn|1|\n", "format": SOURCE},
            )
        return httpx.Response(404)

    from pokecoach_trainer.collection_client import ShowdownReplayClient

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        tmp_path,
        view.job.job_id,
        "dl-ten",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=10,
        clock=clock,
    )
    assert fetched == 10
    client.close()


def test_old_pending_tasks_are_adopted_without_rescan(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    clock = FakeClock()
    store = CollectionStore(tmp_path, clock=clock)
    first = store.start_campaign(mc_config(max_selected_replays=5))
    rid = f"{SOURCE}-pending"
    store.enqueue_replays(
        first.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )
    store.command(first.job.job_id, "stop")
    second = store.start_campaign(mc_config(max_selected_replays=5))
    with catalog_connection(tmp_path) as conn:
        state = conn.execute("SELECT state FROM download_tasks WHERE replay_id=?", (rid,)).fetchone()
        windows = conn.execute("SELECT COUNT(*) AS c FROM windows").fetchone()["c"]
    assert state["state"] == "pending"
    assert windows == 0


def test_discovery_hole_survives_newer_watermark(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path, clock=FakeClock())
    with catalog_connection(tmp_path) as conn:
        conn.execute(
            """
            INSERT INTO coverage(
                segment_id, source_format, start_ts, end_ts, scan_generation,
                proof_kind, proof_payload
            ) VALUES ('old-hole', ?, 100, 200, 0, 'gap', '{}')
            """,
            (SOURCE,),
        )
        conn.execute(
            """
            INSERT INTO coverage(
                segment_id, source_format, start_ts, end_ts, scan_generation,
                proof_kind, proof_payload
            ) VALUES ('new-seg', ?, 900000, 950000, 0, 'scan_complete', '{}')
            """,
            (SOURCE,),
        )
    bounds = incremental_bounds(store, SOURCE, end=1_000_000, overlap_hours=48)
    assert bounds.end == 1_000_000
    with catalog_connection(tmp_path) as conn:
        holes = conn.execute(
            "SELECT COUNT(*) AS c FROM coverage WHERE proof_kind='gap'"
        ).fetchone()["c"]
    assert holes == 1
