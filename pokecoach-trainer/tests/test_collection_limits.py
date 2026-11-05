import random

import pytest

from pokecoach_trainer.collection_limits import acquire_request_permit, record_request_result
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem
from pokecoach_trainer.schemas import CollectionConfig
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
def limit_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    yield tmp_path


def test_global_rate_across_eight_workers(limit_env):
    clock = FakeClock()
    store = CollectionStore(limit_env, clock=clock)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], requests_per_s=10.0, max_requests=100)
    )
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem("gen9championsvgc2026regmc-a", 1, "f")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    times: list[float] = []
    for i in range(4):
        clock._mono = 100.0 + (i + 1) * 0.2
        result = acquire_request_permit(store, claim.lease, "search", jitter=random.Random(0))
        if result.permit:
            times.append(clock.monotonic())
            record_request_result(store, result.permit, status_code=200, elapsed_s=0.1)
    assert len(times) >= 1
    if len(times) >= 2:
        assert times[1] - times[0] >= 0.05


def test_retry_also_consumes_budget(limit_env):
    clock = FakeClock()
    store = CollectionStore(limit_env, clock=clock)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_requests=5, max_selected_replays=10)
    )
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem("gen9championsvgc2026regmc-b", 1, "f")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    permits = 0
    for i in range(8):
        clock._mono = 100.0 + (i + 1) * 2.0
        result = acquire_request_permit(store, claim.lease, "fetch", jitter=random.Random(0))
        if result.permit:
            permits += 1
            record_request_result(store, result.permit, status_code=200, elapsed_s=0.01)
    assert permits == 5


def test_crash_after_permit_does_not_refund_request(limit_env):
    clock = FakeClock()
    store = CollectionStore(limit_env, clock=clock)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_requests=3))
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem("gen9championsvgc2026regmc-crash", 1, "f")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    result = acquire_request_permit(store, claim.lease, "fetch", jitter=random.Random(0))
    assert result.permit
    store2 = CollectionStore(limit_env, clock=clock)
    after = store2.get_campaign(view.job.job_id).counters["requests_consumed"]
    assert after == 1


def test_parallel_new_replay_budget_never_overshoots(limit_env):
    import multiprocessing as mp

    clock = FakeClock()
    store = CollectionStore(limit_env, clock=clock)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=5, max_requests=100)
    )
    items = [
        SearchItem(f"gen9championsvgc2026regmc-p{i}", i, "f") for i in range(20)
    ]
    store.enqueue_replays(view.job.job_id, items)

    ctx = mp.get_context("fork")

    def worker(out: mp.Queue) -> None:
        local = CollectionStore(limit_env, clock=clock)
        selected = 0
        for _ in range(10):
            claim = local.claim_download(view.job.job_id, "w-parallel")
            if not claim.lease:
                continue
            from pokecoach_trainer.collection_filters import evaluate_selection

            decision = evaluate_selection(
                CollectionConfig(format_ids=["mc"]), uploadtime=1, summary=None
            )
            local.record_selection(claim.lease, decision, None)
            if decision.selected:
                selected += 1
        out.put(selected)

    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(q,)) for _ in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=60)
    final = CollectionStore(limit_env, clock=clock).get_campaign(view.job.job_id)
    assert final.counters["selected_in_attempt"] <= 5


def test_deadline_prevents_new_requests(limit_env):
    clock = FakeClock()
    store = CollectionStore(limit_env, clock=clock)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_requests=10, max_duration_s=10)
    )
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem("gen9championsvgc2026regmc-dead", 1, "f")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    clock._wall += 20.0
    result = acquire_request_permit(store, claim.lease, "fetch", jitter=random.Random(0))
    assert result.permit is None
    assert result.stop_reason == "deadline"


def test_retry_after_date_and_seconds_pause_every_worker(limit_env):
    clock = FakeClock()
    store = CollectionStore(limit_env, clock=clock)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_requests=10))
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem("gen9championsvgc2026regmc-429", 1, "f")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    permit = acquire_request_permit(store, claim.lease, "fetch", jitter=random.Random(0))
    assert permit.permit
    record_request_result(store, permit.permit, status_code=429, elapsed_s=0.0, retry_after="2")
    blocked = acquire_request_permit(store, claim.lease, "fetch", jitter=random.Random(0))
    assert blocked.permit is None
    assert blocked.wait_s >= 1.0


def test_invalid_retry_after_uses_backoff(limit_env):
    clock = FakeClock()
    store = CollectionStore(limit_env, clock=clock)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_requests=10))
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem("gen9championsvgc2026regmc-badra", 1, "f")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    permit = acquire_request_permit(store, claim.lease, "fetch", jitter=random.Random(0))
    assert permit.permit
    record_request_result(
        store, permit.permit, status_code=429, elapsed_s=0.0, retry_after="not-a-date"
    )
    limits = store.host_limit_state()
    assert limits["provider_pause_until_wall"] is not None
    assert float(limits["provider_pause_until_wall"]) >= clock.wall()


def test_stale_lease_cannot_take_permit(limit_env):
    clock = FakeClock()
    store = CollectionStore(limit_env, clock=clock)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_requests=10))
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem("gen9championsvgc2026regmc-stale", 1, "f")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    store.release_lease(claim.lease)
    result = acquire_request_permit(store, claim.lease, "fetch", jitter=random.Random(0))
    assert result.permit is None
    assert result.stop_reason == "stale_lease"
