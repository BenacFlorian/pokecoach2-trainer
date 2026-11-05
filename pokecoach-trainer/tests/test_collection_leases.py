"""Tests réservations de tâches collecte."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace

import pytest

from pokecoach_trainer.collection_filters import evaluate_selection
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem, WindowBounds
from pokecoach_trainer.errors import StaleLease
from pokecoach_trainer.schemas import CollectionConfig, TeamFilter
from pokecoach_trainer.settings import clear_settings_cache


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


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    yield tmp_path


def _start(store_root, clock: FakeClock | None = None) -> tuple[CollectionStore, str]:
    store = CollectionStore(store_root, clock=clock or FakeClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=2000))
    return store, view.job.job_id


def test_eight_processes_claim_each_id_once(store_root):
    store, campaign_id = _start(store_root)
    items = [
        SearchItem(replay_id=f"gen9championsvgc2026regmc-{i}", uploadtime=i, source_format="f")
        for i in range(200)
    ]
    store.enqueue_replays(campaign_id, items)

    ctx = mp.get_context("fork")

    def worker(worker_idx: int, out: mp.Queue) -> None:
        clock = FakeClock()
        local = CollectionStore(store_root, clock=clock)
        claimed: list[str] = []
        while True:
            result = local.claim_download(campaign_id, f"w-{worker_idx}")
            if not result.lease:
                break
            claimed.append(result.lease.replay_id or "")
        out.put(claimed)

    q: mp.Queue = ctx.Queue()
    processes = [ctx.Process(target=worker, args=(i, q)) for i in range(8)]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=120)
    all_ids: list[str] = []
    for _ in processes:
        all_ids.extend(q.get(timeout=10))
    assert len(all_ids) == 200
    assert len(all_ids) == len(set(all_ids))


def test_overlapping_discovery_enqueues_once(store_root):
    store, campaign_id = _start(store_root)
    items = [
        SearchItem("gen9championsvgc2026regmc-a", 1, "fmt"),
        SearchItem("gen9championsvgc2026regmc-a", 1, "fmt"),
    ]
    assert store.enqueue_replays(campaign_id, items) == 1


def test_stale_token_cannot_renew_release_or_finish(store_root):
    store, campaign_id = _start(store_root)
    store.enqueue_replays(
        campaign_id,
        [SearchItem("gen9championsvgc2026regmc-x", 1, "fmt")],
    )
    claim = store.claim_download(campaign_id, "w1")
    assert claim.lease
    lease = claim.lease
    with pytest.raises(StaleLease):
        bad = replace(lease, token="wrong")
        store.renew_lease(bad)


def test_rejection_releases_selection_slot_but_not_request_budget(store_root):
    team = TeamFilter(teams=[["pikachu", "eevee", "bulbasaur", "charmander", "squirtle", "mew"]])
    store = CollectionStore(store_root, clock=FakeClock())
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=2000, team_filter=team)
    )
    campaign_id = view.job.job_id
    store.enqueue_replays(
        campaign_id,
        [SearchItem("gen9championsvgc2026regmc-r", 1, "fmt")],
    )
    claim = store.claim_download(campaign_id, "w1")
    assert claim.lease
    attempt_before = store.get_campaign(campaign_id).counters["requests_consumed"]
    decision = evaluate_selection(
        CollectionConfig(format_ids=["mc"], team_filter=team),
        uploadtime=1,
        summary=None,
    )
    store.record_selection(claim.lease, decision, None)
    after = store.get_campaign(campaign_id).counters
    assert after["requests_consumed"] == attempt_before
    assert after["reserved_slots"] == 0


def test_selection_releases_reserved_slot_when_accepted(store_root):
    store, campaign_id = _start(store_root)
    store.enqueue_replays(
        campaign_id,
        [SearchItem("gen9championsvgc2026regmc-ok", 1, "fmt")],
    )
    claim = store.claim_download(campaign_id, "w1")
    assert claim.lease
    assert store.get_campaign(campaign_id).counters["reserved_slots"] == 1
    decision = evaluate_selection(CollectionConfig(format_ids=["mc"]), uploadtime=1, summary=None)
    store.record_selection(claim.lease, decision, None)
    after = store.get_campaign(campaign_id).counters
    assert after["selected_in_attempt"] == 1
    assert after["reserved_slots"] == 0


def test_reconcile_reserved_slots_unblocks_pending_after_drift(store_root):
    """selected compté sans libérer reserved ne doit pas bloquer la file pending."""

    store = CollectionStore(store_root, clock=FakeClock())
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=100_000)
    )
    campaign_id = view.job.job_id
    items = [SearchItem(f"gen9championsvgc2026regmc-p{i}", i, "fmt") for i in range(3)]
    store.enqueue_replays(campaign_id, items)
    claim = store.claim_download(campaign_id, "w1")
    assert claim.lease
    decision = evaluate_selection(CollectionConfig(format_ids=["mc"]), uploadtime=1, summary=None)
    store.record_selection(claim.lease, decision, None)
    with store._tx() as conn:
        conn.execute(
            "UPDATE attempts SET reserved_slots = reserved_slots + 50000 WHERE attempt_id=?",
            (claim.lease.attempt_id,),
        )
    claim2 = store.claim_download(campaign_id, "w2")
    assert claim2.lease is not None
    assert claim2.reason is None


def test_attempts_survive_expiry(store_root):
    clock = FakeClock()
    store, campaign_id = _start(store_root, clock)
    store.enqueue_replays(
        campaign_id,
        [SearchItem("gen9championsvgc2026regmc-exp", 1, "fmt")],
    )
    claim = store.claim_download(campaign_id, "w1")
    assert claim.lease
    store.reserve_request_permit(
        attempt_id=claim.lease.attempt_id,
        task_id=claim.lease.task_id,
        kind="fetch",
        permit_id="perm-1",
    )
    clock._wall += 500.0
    store.reconcile_expired_download_leases(campaign_id)
    again = store.claim_download(campaign_id, "w2")
    assert again.lease
    assert again.lease.attempts_used >= 1


def test_boot_change_expires_old_lease(store_root):
    clock = FakeClock()
    store, campaign_id = _start(store_root, clock)
    store.enqueue_replays(
        campaign_id,
        [SearchItem("gen9championsvgc2026regmc-boot", 1, "fmt")],
    )
    claim = store.claim_download(campaign_id, "w1")
    assert claim.lease
    clock._boot = "boot-b"
    with pytest.raises(StaleLease):
        store.renew_lease(claim.lease)
    other = store.claim_download(campaign_id, "w2")
    assert other.lease
    assert other.lease.replay_id == claim.lease.replay_id


def test_clock_rollback_does_not_create_second_owner(store_root):
    clock = FakeClock()
    store, campaign_id = _start(store_root, clock)
    bounds = WindowBounds(source_format="gen9championsvgc2026regmc", start=100, end=200)
    store.ensure_window(campaign_id, bounds, 0)
    first = store.claim_window(campaign_id, "disc-a")
    assert first.lease
    clock._mono = 50.0
    second = store.claim_window(campaign_id, "disc-b")
    assert second.lease is None
