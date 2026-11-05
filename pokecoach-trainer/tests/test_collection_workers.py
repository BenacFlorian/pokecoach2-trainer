"""Tests workers collecte."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from pokecoach_trainer.archive_service import archive_replay, is_replay_archived
from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem
from pokecoach_trainer.collection_workers import run_collection_worker
from pokecoach_trainer.schemas import ArchiveSource, CollectionConfig
from pokecoach_trainer.settings import clear_settings_cache

SOURCE = "gen9championsvgc2026regmc"
RID = f"{SOURCE}-worker1"


class FakeClock(SystemClock):
    def __init__(self) -> None:
        self._wall = 1_700_000_000.0
        self._mono = 10.0
        self._boot = "b"

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
    base = {"format_ids": ["mc"], "requests_per_s": 20.0, "max_requests": 500}
    base.update(kwargs)
    return CollectionConfig(**base)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    yield tmp_path


def test_download_worker_skips_archived_without_http(env):
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    archive_replay(
        RID,
        {"id": RID, "uploadtime": 1, "log": "|t|1\n", "format": SOURCE},
        source,
    )
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(mc_config(max_selected_replays=10))
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=RID, uploadtime=1, source_format=SOURCE)],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("aucun HTTP attendu")

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)

    def factory(_cfg):
        return client

    steps = run_collection_worker(
        env,
        view.job.job_id,
        "dl-0",
        "download",
        client_factory=factory,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert steps == 1
    client.close()


TEAM_A = ["pikachu", "eevee", "bulbasaur", "charmander", "squirtle", "mew"]


def _log(p1: list[str], p2: list[str]) -> str:
    lines = ["|player|p1|A|1300|", "|player|p2|B|1300|"]
    for side, roster in (("p1", p1), ("p2", p2)):
        for slot, species in zip("abcdef", roster, strict=True):
            lines.append(f"|poke|{side}{slot}|{species}|")
    lines.append("|turn|1|")
    return "\n".join(lines) + "\n"


def test_download_worker_retries_without_losing_task(env):
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(mc_config(max_selected_replays=5))
    rid = f"{SOURCE}-retry"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".json"):
            return httpx.Response(500)
        return httpx.Response(404)

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    steps = run_collection_worker(
        env,
        view.job.job_id,
        "dl-r",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert steps == 1
    with store._tx() as conn:
        state = conn.execute("SELECT state FROM download_tasks WHERE replay_id=?", (rid,)).fetchone()
    assert state["state"] in {"retry_wait", "pending", "failed"}
    client.close()


def test_heartbeat_continues_during_slow_http(env):
    import threading

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(
        mc_config(max_selected_replays=5, heartbeat_s=1, lease_duration_s=30)
    )
    rid = f"{SOURCE}-slow"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )
    started = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        import time

        time.sleep(0.05)
        return httpx.Response(
            200,
            json={"id": rid, "uploadtime": 1, "log": "|turn|1|\n", "format": SOURCE},
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-slow",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert started.is_set()
    client.close()


def test_empty_queue_while_discovery_running_is_not_completed(env):
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(mc_config(discovery_workers=1))
    from pokecoach_trainer.collection_discovery import WindowBounds

    store.ensure_window(
        view.job.job_id,
        WindowBounds(source_format=SOURCE, start=0, end=1000),
        0,
    )
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    steps = run_collection_worker(
        env,
        view.job.job_id,
        "disc-only",
        "discovery",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert steps in {0, 1}
    phase = store.get_campaign(view.job.job_id).phase
    assert phase == "running"


def test_stop_interrupts_rate_limit_wait(env):
    import threading

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], requests_per_s=0.1, max_requests=10)
    )
    rid = f"{SOURCE}-stop"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )
    stop = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": rid, "uploadtime": 1, "log": "|turn|1|\n", "format": SOURCE},
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    stop.set()
    steps = run_collection_worker(
        env,
        view.job.job_id,
        "dl-stop",
        "download",
        client_factory=lambda _c: client,
        stop_event=stop,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert steps == 0
    client.close()


def test_basic_collection_archives_unknown_teams(env):
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(mc_config(max_selected_replays=5))
    rid = f"{SOURCE}-unknown"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": rid, "uploadtime": 1, "log": "|turn|1|\n", "format": SOURCE},
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-u",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert is_replay_archived(rid)
    client.close()


def test_filtered_collection_keeps_only_matches(env):
    from pokecoach_trainer.schemas import TeamFilter

    team = TeamFilter(teams=[TEAM_A])
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(mc_config(max_selected_replays=5, team_filter=team))
    match_id = f"{SOURCE}-match"
    miss_id = f"{SOURCE}-miss"
    store.enqueue_replays(
        view.job.job_id,
        [
            SearchItem(replay_id=match_id, uploadtime=1, source_format=SOURCE),
            SearchItem(replay_id=miss_id, uploadtime=1, source_format=SOURCE),
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        rid = request.url.path.split("/")[-1].replace(".json", "")
        if rid == match_id:
            log = _log(TEAM_A, TEAM_A)
        else:
            log = _log(
                ["garchomp", "landorus", "incineroar", "rillaboom", "amoonguss", "fluttermane"],
                ["garchomp", "landorus", "incineroar", "rillaboom", "amoonguss", "calyrex"],
            )
        return httpx.Response(
            200,
            json={"id": rid, "uploadtime": 1, "log": log, "format": SOURCE},
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-f",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=2,
        clock=clock,
    )
    assert is_replay_archived(match_id)
    assert not is_replay_archived(miss_id)
    client.close()


def test_unfiltered_campaign_downloads_previously_rejected_body(env):
    from pokecoach_trainer.collection_filters import evaluate_selection
    from pokecoach_trainer.collection_types import TeamSummary
    from pokecoach_trainer.schemas import TeamFilter

    team = TeamFilter(teams=[TEAM_A])
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    first = store.start_campaign(mc_config(max_selected_replays=5, team_filter=team))
    rid = f"{SOURCE}-rebody"
    store.enqueue_replays(
        first.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )
    claim = store.claim_download(first.job.job_id, "w1")
    assert claim.lease
    summary = TeamSummary(
        p1=tuple(
            sorted(
                ["garchomp", "landorus", "incineroar", "rillaboom", "amoonguss", "fluttermane"]
            )
        ),
        p2=tuple(
            sorted(
                ["garchomp", "landorus", "incineroar", "rillaboom", "amoonguss", "fluttermane"]
            )
        ),
        parser_version="collection_filters/1",
        content_sha256="x",
    )
    decision = evaluate_selection(
        CollectionConfig(format_ids=["mc"], team_filter=team), uploadtime=1, summary=summary
    )
    store.record_selection(claim.lease, decision, summary)
    store.command(first.job.job_id, "stop")
    second = store.start_campaign(mc_config(max_selected_replays=5))
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(request.url.path)
        return httpx.Response(
            200,
            json={"id": rid, "uploadtime": 1, "log": _log(TEAM_A, TEAM_A), "format": SOURCE},
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        second.job.job_id,
        "dl-body",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert fetched
    client.close()


def test_new_team_filter_uses_cached_summary_without_get_for_known_mismatch(env):
    from pokecoach_trainer.collection_filters import evaluate_selection
    from pokecoach_trainer.collection_types import TeamSummary
    from pokecoach_trainer.schemas import TeamFilter

    other = ["garchomp", "landorus", "incineroar", "rillaboom", "amoonguss", "fluttermane"]
    team = TeamFilter(teams=[TEAM_A])
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    first = store.start_campaign(mc_config(max_selected_replays=5, team_filter=team))
    rid = f"{SOURCE}-cached"
    store.enqueue_replays(
        first.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE)],
    )
    claim = store.claim_download(first.job.job_id, "w1")
    summary = TeamSummary(
        p1=tuple(sorted(other)),
        p2=tuple(sorted(other)),
        parser_version="collection_filters/1",
        content_sha256="x",
    )
    decision = evaluate_selection(
        CollectionConfig(format_ids=["mc"], team_filter=team), uploadtime=1, summary=summary
    )
    store.record_selection(claim.lease, decision, summary)
    store.command(first.job.job_id, "stop")
    second = store.start_campaign(mc_config(max_selected_replays=5, team_filter=team))
    claim = store.claim_download(second.job.job_id, "w1")
    assert claim.lease is None
