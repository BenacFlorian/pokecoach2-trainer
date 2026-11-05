"""Tests découverte par fenêtres."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from pokecoach_trainer.collection_discovery import partition_range, process_search_page
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import DiscoveryPage, SearchItem, TaskLease, WindowBounds
from pokecoach_trainer.schemas import CollectionConfig
from pokecoach_trainer.settings import clear_settings_cache

DAY = 86_400


class FakeClock(SystemClock):
    def __init__(self) -> None:
        self._wall = 2_000_000_000.0
        self._mono = 500.0
        self._boot = "boot-a"

    def wall(self) -> float:
        return self._wall

    def monotonic(self) -> float:
        return self._mono

    def boot_id(self) -> str:
        return self._boot


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    yield tmp_path


def test_windows_are_disjoint_on_utc_grid():
    start = 100
    end = 2_000_000
    windows = list(partition_range("fmt", start, end, partition_days=7))
    assert windows
    assert windows[0].start == start
    assert windows[-1].end == end
    for left, right in zip(windows, windows[1:]):
        assert left.end == right.start
        assert left.end - left.start <= 7 * DAY


def test_two_discovery_workers_do_not_scan_same_window(store_root):
    clock = FakeClock()
    store = CollectionStore(store_root, clock=clock)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=100))
    campaign_id = view.job.job_id
    bounds = WindowBounds(source_format="gen9championsvgc2026regmc", start=1, end=1000)
    store.ensure_window(campaign_id, bounds, 0)
    barrier = mp.Barrier(2)

    def worker(wid: str, out: mp.Queue) -> None:
        local = CollectionStore(store_root, clock=clock)
        barrier.wait()
        result = local.claim_window(campaign_id, wid)
        out.put(result.lease.task_id if result.lease else None)

    q: mp.Queue = mp.Queue()
    ps = [mp.Process(target=worker, args=(f"w{i}", q)) for i in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=30)
    ids = [q.get(timeout=5) for _ in ps]
    claimed = [x for x in ids if x]
    assert len(claimed) == len(set(claimed))
    assert len(claimed) == 1


def test_page_ids_and_cursor_commit_together(store_root):
    clock = FakeClock()
    store = CollectionStore(store_root, clock=clock)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=50))
    campaign_id = view.job.job_id
    bounds = WindowBounds(source_format="gen9championsvgc2026regmc", start=0, end=10_000)
    store.ensure_window(campaign_id, bounds, 0)
    claim = store.claim_window(campaign_id, "disc-1")
    assert claim.lease
    lease = claim.lease
    page = DiscoveryPage(
        items=(
            SearchItem(
                replay_id="gen9championsvgc2026regmc-a",
                uploadtime=5000,
                source_format="gen9championsvgc2026regmc",
            ),
        ),
        next_before=4000,
        exhausted=False,
        blocked_reason=None,
    )
    process_search_page(store, lease, page)
    with store._tx() as conn:
        row = conn.execute(
            "SELECT cursor_before FROM windows WHERE window_id=?", (lease.task_id,)
        ).fetchone()
        assert int(row["cursor_before"]) == 4000
        n = conn.execute("SELECT COUNT(*) AS c FROM download_tasks").fetchone()["c"]
        assert n == 1


def test_same_timestamp_stall_is_reported(store_root):
    store = CollectionStore(store_root, clock=FakeClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    bounds = WindowBounds(source_format="gen9championsvgc2026regmc", start=0, end=5000)
    store.ensure_window(view.job.job_id, bounds, 0)
    claim = store.claim_window(view.job.job_id, "d1")
    lease = claim.lease
    assert lease
    page = DiscoveryPage(
        items=tuple(),
        next_before=lease.cursor_before,
        exhausted=False,
        blocked_reason=None,
    )
    process_search_page(store, lease, page)
    with store._tx() as conn:
        state = conn.execute("SELECT state FROM windows LIMIT 1").fetchone()["state"]
        assert state == "blocked"


def test_out_of_window_id_is_not_downloaded(store_root):
    store = CollectionStore(store_root, clock=FakeClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    bounds = WindowBounds(source_format="gen9championsvgc2026regmc", start=100, end=200)
    store.ensure_window(view.job.job_id, bounds, 0)
    claim = store.claim_window(view.job.job_id, "d1")
    lease = claim.lease
    assert lease
    page = DiscoveryPage(
        items=(
            SearchItem(
                replay_id="gen9championsvgc2026regmc-x",
                uploadtime=50,
                source_format="gen9championsvgc2026regmc",
            ),
        ),
        next_before=99,
        exhausted=False,
        blocked_reason=None,
    )
    process_search_page(store, lease, page)
    with store._tx() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM download_tasks").fetchone()["c"]
        assert n == 0


def test_empty_middle_window_does_not_end_history(store_root):
    store = CollectionStore(store_root, clock=FakeClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    bounds = WindowBounds(source_format="gen9championsvgc2026regmc", start=1_000, end=2_000)
    store.ensure_window(view.job.job_id, bounds, 0)
    claim = store.claim_window(view.job.job_id, "d1")
    lease = claim.lease
    assert lease
    page = DiscoveryPage(
        items=(
            SearchItem(
                replay_id="gen9championsvgc2026regmc-x",
                uploadtime=500,
                source_format="gen9championsvgc2026regmc",
            ),
        ),
        next_before=1_500,
        exhausted=False,
        blocked_reason=None,
    )
    process_search_page(store, lease, page)
    with store._tx() as conn:
        state = conn.execute("SELECT state FROM windows LIMIT 1").fetchone()["state"]
        proof = conn.execute(
            "SELECT COUNT(*) AS c FROM coverage WHERE proof_kind='empty_search'"
        ).fetchone()["c"]
    assert state == "scanning"
    assert proof == 0


def test_partial_budget_does_not_complete_window(store_root):
    clock = FakeClock()
    store = CollectionStore(store_root, clock=clock)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_requests=1, max_selected_replays=10)
    )
    bounds = WindowBounds(source_format="gen9championsvgc2026regmc", start=100, end=200)
    store.ensure_window(view.job.job_id, bounds, 0)
    claim = store.claim_window(view.job.job_id, "d1")
    lease = claim.lease
    assert lease
    store.reserve_request_permit(
        attempt_id=lease.attempt_id,
        task_id=lease.task_id,
        kind="search",
        permit_id="used",
    )
    page = DiscoveryPage(
        items=tuple(),
        next_before=50,
        exhausted=False,
        blocked_reason=None,
    )
    process_search_page(store, lease, page)
    with store._tx() as conn:
        state = conn.execute("SELECT state FROM windows LIMIT 1").fetchone()["state"]
    assert state == "pending"


def test_ten_successful_pages_are_not_limited_to_five_attempts(store_root):
    clock = FakeClock()
    store = CollectionStore(store_root, clock=clock)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_attempts=5, max_selected_replays=50)
    )
    bounds = WindowBounds(source_format="gen9championsvgc2026regmc", start=0, end=50_000)
    store.ensure_window(view.job.job_id, bounds, 0)
    cursor = 40_000
    claim = store.claim_window(view.job.job_id, "disc-main")
    assert claim.lease
    lease = claim.lease
    for step in range(10):
        nxt = cursor - 500
        page = DiscoveryPage(
            items=(
                SearchItem(
                    replay_id=f"gen9championsvgc2026regmc-{step}",
                    uploadtime=cursor - 100,
                    source_format="gen9championsvgc2026regmc",
                ),
            ),
            next_before=nxt,
            exhausted=False,
            blocked_reason=None,
        )
        process_search_page(store, lease, page)
        cursor = nxt
        with store._tx() as conn:
            cur = conn.execute(
                "SELECT cursor_before FROM windows WHERE window_id=?", (lease.task_id,)
            ).fetchone()
        lease = TaskLease(
            kind=lease.kind,
            task_id=lease.task_id,
            campaign_id=lease.campaign_id,
            attempt_id=lease.attempt_id,
            worker_id=lease.worker_id,
            token=lease.token,
            expires_at=lease.expires_at,
            replay_id=lease.replay_id,
            window=lease.window,
            cursor_before=int(cur["cursor_before"]),
            attempts_used=lease.attempts_used,
        )
    with store._tx() as conn:
        state = conn.execute("SELECT state FROM windows LIMIT 1").fetchone()["state"]
    assert state == "scanning"
    store.release_lease(lease)
    claim = store.claim_window(view.job.job_id, "blocker")
    assert claim.lease
    for _ in range(3):
        store.release_lease(claim.lease)
        again = store.claim_window(view.job.job_id, "blocker")
        assert again.lease
        claim = again
    store.release_lease(claim.lease)
    with store._tx() as conn:
        state = conn.execute("SELECT state FROM windows LIMIT 1").fetchone()["state"]
    assert state == "blocked"
    blocked_claim = store.claim_window(view.job.job_id, "blocker")
    assert blocked_claim.lease is None
