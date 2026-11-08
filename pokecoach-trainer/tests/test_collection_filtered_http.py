"""Scénario HTTP local : quota=3, mélange, reprise, snapshot."""

from __future__ import annotations

import threading

import pytest

from collection_http_server import _Handler, start_server
from pokecoach_trainer.archive_catalog import catalog_connection
from pokecoach_trainer.archive_service import create_campaign_snapshot, is_replay_archived
from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem
from pokecoach_trainer.collection_workers import run_collection_worker
from pokecoach_trainer.schemas import CollectionConfig, ReplayFilterConfig
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout
from replay_log_fixtures import build_log

SOURCE = "gen9championsvgc2026regmc"


class FakeClock(SystemClock):
    def __init__(self) -> None:
        self._wall = 1_700_000_000.0
        self._mono = 10.0
        self._boot = "http-boot"

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


@pytest.fixture
def http_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    _Handler.pages = {}
    server, port = start_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, server, port
    server.shutdown()
    thread.join(timeout=5)
    _Handler.pages = {}


def _payload(rid: str, log: str) -> dict:
    return {"id": rid, "uploadtime": 1, "log": log, "format": SOURCE}


def test_http_quota_three_interrupt_resume_snapshot(http_env):
    root, server, port = http_env
    _ = server
    valid = [
        f"{SOURCE}-ok{i}"
        for i in range(3)
    ]
    rejected = [
        f"{SOURCE}-noots",
        f"{SOURCE}-lowelo",
        f"{SOURCE}-inact",
    ]
    pages: dict[str, object] = {
        valid[0]: _payload(
            valid[0], build_log(p1_name="A0", p1_rating=1300, p2_rating=900)
        ),
        valid[1]: _payload(
            valid[1], build_log(p1_name="A1", p1_rating=1400, p2_rating=1400)
        ),
        valid[2]: _payload(
            valid[2], build_log(p1_name="A2", p1_rating=1100, p2_rating=1000)
        ),
        rejected[0]: _payload(
            rejected[0],
            build_log(include_ots=False, p1_rating=1300, p2_rating=1300),
        ),
        rejected[1]: _payload(rejected[1], build_log(p1_rating=900, p2_rating=900)),
        rejected[2]: _payload(
            rejected[2],
            build_log(p1_rating=1300, p2_rating=1300, termination="inactivity"),
        ),
    }
    _Handler.pages = pages  # type: ignore[assignment]
    clock = FakeClock()
    store = CollectionStore(root, clock=clock)
    view = store.start_campaign(
        CollectionConfig(
            format_ids=["mc"],
            max_selected_replays=3,
            requests_per_s=20.0,
            max_requests=50,
            dataset_filters=ReplayFilterConfig(min_completed_turns_forfeit=2),
        )
    )
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(rid, 1, SOURCE) for rid in [*rejected, *valid]],
    )
    site = f"http://127.0.0.1:{port}"

    def factory(_cfg):
        return ShowdownReplayClient(site=site, sleep=lambda _: None, max_attempts=1)

    run_collection_worker(
        root,
        view.job.job_id,
        "dl-1",
        "download",
        client_factory=factory,
        sleep=tick(clock),
        max_steps=2,
        clock=clock,
    )
    store.command(view.job.job_id, "pause")
    store.command(view.job.job_id, "resume")
    for worker_id in ("dl-a", "dl-b"):
        run_collection_worker(
            root,
            view.job.job_id,
            worker_id,
            "download",
            client_factory=factory,
            sleep=tick(clock),
            max_steps=6,
            clock=clock,
        )
    campaign = store.get_campaign(view.job.job_id)
    assert campaign.counters["selected_in_attempt"] == 3
    for rid in valid:
        assert is_replay_archived(rid)
    for rid in rejected:
        assert not is_replay_archived(rid)
    with catalog_connection(root) as conn:
        selected = {
            row["replay_id"]
            for row in conn.execute(
                "SELECT replay_id FROM campaign_replays WHERE decision='selected'"
            )
        }
        rejected_n = conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_replays WHERE decision='rejected'"
        ).fetchone()["n"]
    assert selected == set(valid)
    assert rejected_n == 3
    snap = create_campaign_snapshot(view.job.job_id)
    assert len(snap.refs) == 3
    assert {ref.replay_id for ref in snap.refs} == set(valid)
