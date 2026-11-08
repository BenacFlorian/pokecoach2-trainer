"""Tests campagnes et tentatives de collecte."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from pokecoach_trainer.archive_catalog import catalog_connection
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem
from pokecoach_trainer.errors import Conflict
from pokecoach_trainer.schemas import CollectionBudgets, CollectionConfig
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


def test_simultaneous_identical_start_without_upper_bound_returns_one_campaign(store_root):
    barrier = mp.Barrier(2)

    def worker(result_queue: mp.Queue) -> None:
        barrier.wait()
        store = CollectionStore(store_root)
        view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
        result_queue.put(view.job.job_id)

    q: mp.Queue = mp.Queue()
    p1 = mp.Process(target=worker, args=(q,))
    p2 = mp.Process(target=worker, args=(q,))
    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)
    ids = {q.get(timeout=5), q.get(timeout=5)}
    assert len(ids) == 1


def test_different_active_config_conflicts(store_root):
    store = CollectionStore(store_root)
    store.start_campaign(CollectionConfig(format_ids=["mc"]))
    with pytest.raises(Conflict):
        store.start_campaign(CollectionConfig(format_ids=["mc"], mode="incremental"))


def test_pause_resume_keeps_attempt_budget(store_root):
    store = CollectionStore(store_root)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=10))
    before = view.counters["selected_in_attempt"]
    store.command(view.job.job_id, "pause")
    resumed = store.command(view.job.job_id, "resume")
    assert resumed.counters["selected_in_attempt"] == before
    assert resumed.attempt_id == view.attempt_id


def test_new_attempt_requires_explicit_budgets(store_root):
    store = CollectionStore(store_root)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=1, max_requests=1)
    )
    with catalog_connection(store_root) as conn:
        conn.execute(
            "UPDATE campaigns SET phase='budget_reached' WHERE campaign_id=?",
            (view.job.job_id,),
        )
    with pytest.raises(Conflict):
        store.command(view.job.job_id, "resume")
    budgets = CollectionBudgets(max_selected_replays=5, max_requests=5, max_duration_s=60)
    new_view = store.command(view.job.job_id, "resume", new_budgets=budgets)
    assert new_view.attempt_id != view.attempt_id


TEAM_A = ["pikachu", "eevee", "bulbasaur", "charmander", "squirtle", "mew"]
SOURCE = "gen9championsvgc2026regmc"
REFILTER_ID = f"{SOURCE}-refilter"


def test_new_filter_revisits_prior_rejections_without_rediscovery(store_root):
    from pokecoach_trainer.collection_filters import evaluate_selection
    from pokecoach_trainer.collection_types import TeamSummary
    from pokecoach_trainer.schemas import TeamFilter

    clock = FakeClock()
    team_filter = TeamFilter(teams=[TEAM_A])
    store = CollectionStore(store_root, clock=clock)
    first = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=10, team_filter=team_filter)
    )
    store.enqueue_replays(
        first.job.job_id,
        [SearchItem(replay_id=REFILTER_ID, uploadtime=5_000, source_format=SOURCE)],
    )
    claim = store.claim_download(first.job.job_id, "w1")
    assert claim.lease
    other = ["garchomp", "landorus", "incineroar", "rillaboom", "amoonguss", "fluttermane"]
    summary = TeamSummary(
        p1=tuple(sorted(other)),
        p2=tuple(sorted(other)),
        parser_version="collection_filters/1",
        content_sha256="abc",
    )
    decision = evaluate_selection(
        CollectionConfig(format_ids=["mc"], team_filter=team_filter),
        uploadtime=5_000,
        summary=summary,
    )
    store.record_selection(claim.lease, decision, summary)
    store.command(first.job.job_id, "stop")
    with catalog_connection(store_root) as conn:
        windows_before = conn.execute("SELECT COUNT(*) AS c FROM windows").fetchone()["c"]
    store2 = CollectionStore(store_root, clock=clock)
    second = store2.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=10))
    with catalog_connection(store_root) as conn:
        windows_after = conn.execute("SELECT COUNT(*) AS c FROM windows").fetchone()["c"]
        task = conn.execute(
            "SELECT state FROM download_tasks WHERE replay_id=?", (REFILTER_ID,)
        ).fetchone()
        row = conn.execute(
            """
            SELECT decision FROM campaign_replays
            WHERE campaign_id=? AND replay_id=?
            """,
            (second.job.job_id, REFILTER_ID),
        ).fetchone()
    assert windows_after == windows_before
    assert task["state"] == "pending"
    assert row is None


def test_modified_team_file_does_not_change_resume(store_root):
    from pokecoach_trainer.schemas import TeamFilter

    clock = FakeClock()
    team_a = TeamFilter(teams=[TEAM_A])
    team_b = TeamFilter(
        teams=[["garchomp", "landorus", "incineroar", "rillaboom", "amoonguss", "fluttermane"]]
    )
    store = CollectionStore(store_root, clock=clock)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=5, team_filter=team_a)
    )
    frozen = view.effective_config.team_filter
    store.command(view.job.job_id, "pause")
    _ = team_b  # contenu alternatif simulé, non rechargé à la reprise
    resumed = store.command(view.job.job_id, "resume")
    assert resumed.effective_config.team_filter == frozen
    assert resumed.effective_config.team_filter != team_b
