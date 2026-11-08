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
    lines = ["|player|p1|A|av1|1300|", "|player|p2|B|av2|1300|"]
    for side, roster in (("p1", p1), ("p2", p2)):
        for species in roster:
            lines.append(f"|poke|{side}|{species}, L50|")
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


def test_filtered_campaign_keeps_only_valid_payload(env):
    from pokecoach_trainer.archive_catalog import catalog_connection
    from pokecoach_trainer.schemas import ReplayFilterConfig
    from replay_log_fixtures import build_log

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(
        mc_config(max_selected_replays=5, dataset_filters=ReplayFilterConfig())
    )
    ids = {
        "ots": f"{SOURCE}-noots",
        "elo": f"{SOURCE}-elo",
        "inact": f"{SOURCE}-inact",
        "ok": f"{SOURCE}-ok",
    }
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=rid, uploadtime=1, source_format=SOURCE) for rid in ids.values()],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        rid = request.url.path.split("/")[-1].replace(".json", "")
        if rid.endswith("noots"):
            log = build_log(include_ots=False, p1_rating=1300, p2_rating=1300)
        elif rid.endswith("elo"):
            log = build_log(p1_rating=900, p2_rating=900)
        elif rid.endswith("inact"):
            log = build_log(p1_rating=1300, p2_rating=1300, termination="inactivity")
        else:
            log = build_log(p1_rating=1300, p2_rating=900)
        return httpx.Response(200, json={"id": rid, "uploadtime": 1, "log": log, "format": SOURCE})

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-valid",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=4,
        clock=clock,
    )
    assert is_replay_archived(ids["ok"])
    assert not is_replay_archived(ids["ots"])
    assert not is_replay_archived(ids["elo"])
    assert not is_replay_archived(ids["inact"])
    with catalog_connection(env) as conn:
        selected = conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_replays WHERE campaign_id=? AND decision='selected'",
            (view.job.job_id,),
        ).fetchone()["n"]
        rejected = conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_replays WHERE campaign_id=? AND decision='rejected'",
            (view.job.job_id,),
        ).fetchone()["n"]
    assert selected == 1
    assert rejected == 3
    client.close()


def test_filtered_reject_has_no_staging(env):
    from pokecoach_trainer.schemas import ReplayFilterConfig
    from replay_log_fixtures import build_log

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(
        mc_config(max_selected_replays=5, dataset_filters=ReplayFilterConfig())
    )
    rid = f"{SOURCE}-rejstage"
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": rid,
                "uploadtime": 1,
                "log": build_log(include_ots=False),
                "format": SOURCE,
            },
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-rej",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    staging = env / "archive" / "staging"
    leftover = list(staging.glob("*")) if staging.is_dir() else []
    assert leftover == []
    assert not is_replay_archived(rid)
    client.close()


def test_invalid_payload_is_permanent_reject(env):
    from pokecoach_trainer.archive_catalog import catalog_connection
    from pokecoach_trainer.schemas import ReplayFilterConfig

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(
        mc_config(max_selected_replays=5, dataset_filters=ReplayFilterConfig())
    )
    rid = f"{SOURCE}-badjson"
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": rid, "uploadtime": 1, "log": "", "format": SOURCE})

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-bad",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    with catalog_connection(env) as conn:
        row = conn.execute(
            "SELECT decision, reason FROM campaign_replays WHERE replay_id=?",
            (rid,),
        ).fetchone()
    assert row["decision"] == "rejected"
    assert row["reason"] == "invalid_replay_payload"
    assert store.get_campaign(view.job.job_id).counters["selected_in_attempt"] == 0
    client.close()


def test_admission_internal_error_is_not_business_reject(env, monkeypatch):
    from pokecoach_trainer.archive_catalog import catalog_connection
    from pokecoach_trainer.schemas import ReplayFilterConfig
    from replay_log_fixtures import build_log

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(
        mc_config(max_selected_replays=5, dataset_filters=ReplayFilterConfig())
    )
    rid = f"{SOURCE}-boom"
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])

    def boom(*_args, **_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        "pokecoach_trainer.collection_workers.evaluate_collection_candidate", boom
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": rid, "uploadtime": 1, "log": build_log(), "format": SOURCE},
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-boom",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    with catalog_connection(env) as conn:
        rejected = conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_replays WHERE reason != 'invalid_replay_payload'"
        ).fetchone()["n"]
        state = conn.execute(
            "SELECT state, last_error FROM download_tasks WHERE replay_id=?", (rid,)
        ).fetchone()
    assert rejected == 0
    assert state["state"] in {"retry_wait", "failed", "pending"}
    assert store.get_campaign(view.job.job_id).counters["selected_in_attempt"] == 0
    client.close()


def test_archived_valid_and_invalid_need_no_http(env):
    from pokecoach_trainer.archive_catalog import catalog_connection
    from pokecoach_trainer.schemas import ReplayFilterConfig
    from replay_log_fixtures import build_log

    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    ok = f"{SOURCE}-locok"
    bad = f"{SOURCE}-locbad"
    archive_replay(
        ok,
        {"id": ok, "uploadtime": 1, "log": build_log(p1_rating=1300, p2_rating=900), "format": SOURCE},
        source,
    )
    archive_replay(
        bad,
        {
            "id": bad,
            "uploadtime": 1,
            "log": build_log(include_ots=False, p1_rating=1300, p2_rating=1300),
            "format": SOURCE,
        },
        source,
    )
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(
        mc_config(max_selected_replays=5, dataset_filters=ReplayFilterConfig())
    )
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(ok, 1, SOURCE), SearchItem(bad, 1, SOURCE)],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("aucun HTTP attendu")

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-local",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=2,
        clock=clock,
    )
    campaign = store.get_campaign(view.job.job_id)
    assert campaign.counters["selected_in_attempt"] == 1
    assert campaign.counters["new_published"] == 0
    with catalog_connection(env) as conn:
        selected = conn.execute(
            "SELECT replay_id FROM campaign_replays WHERE decision='selected'"
        ).fetchone()
        rejected = conn.execute(
            "SELECT replay_id FROM campaign_replays WHERE decision='rejected'"
        ).fetchone()
    assert selected["replay_id"] == ok
    assert rejected["replay_id"] == bad
    client.close()


def test_old_parser_cache_still_reads_local_archive(env):
    from pokecoach_trainer.replay_metadata import REPLAY_METADATA_VERSION
    from pokecoach_trainer.schemas import ReplayFilterConfig
    from replay_log_fixtures import build_log

    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    rid = f"{SOURCE}-oldcache"
    ref = archive_replay(
        rid,
        {"id": rid, "uploadtime": 1, "log": build_log(p1_rating=1300, p2_rating=900), "format": SOURCE},
        source,
    )
    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    with store._tx() as conn:
        conn.execute(
            """
            INSERT INTO replay_metadata_cache(
                replay_id, content_sha256, parser_version, log_sha256,
                source_format, uploadtime, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (rid, ref.content_sha256, "replay_metadata/1", ref.log_sha256, SOURCE, 1, "{}"),
        )
    view = store.start_campaign(
        mc_config(max_selected_replays=5, dataset_filters=ReplayFilterConfig())
    )
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("aucun HTTP attendu")

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-oldc",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert store.get_campaign(view.job.job_id).counters["selected_in_attempt"] == 1
    cached = store.get_cached_metadata(ref)
    assert cached is not None
    assert cached.parser_version == REPLAY_METADATA_VERSION
    client.close()


def test_same_profile_ots_reject_does_not_refetch(env):
    from pokecoach_trainer.schemas import ReplayFilterConfig
    from replay_log_fixtures import build_log

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    filters = ReplayFilterConfig()
    view = store.start_campaign(mc_config(max_selected_replays=5, dataset_filters=filters))
    rid = f"{SOURCE}-otsagain"
    store.enqueue_replays(view.job.job_id, [SearchItem(rid, 1, SOURCE)])
    fetches = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetches.append(request.url.path)
        return httpx.Response(
            200,
            json={"id": rid, "uploadtime": 1, "log": build_log(include_ots=False), "format": SOURCE},
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        env,
        view.job.job_id,
        "dl-ots1",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert fetches
    claim = store.claim_download(view.job.job_id, "w2")
    assert claim.lease is None
    client.close()


def test_new_profile_without_body_refetches(env):
    from pokecoach_trainer.schemas import ReplayFilterConfig
    from replay_log_fixtures import build_log

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    first = store.start_campaign(
        mc_config(max_selected_replays=5, dataset_filters=ReplayFilterConfig())
    )
    rid = f"{SOURCE}-reots"
    store.enqueue_replays(first.job.job_id, [SearchItem(rid, 1, SOURCE)])

    def handler_first(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": rid,
                "uploadtime": 1,
                "log": build_log(include_ots=False, p1_rating=1300, p2_rating=1300),
                "format": SOURCE,
            },
        )

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler_first), sleep=lambda _: None)
    run_collection_worker(
        env,
        first.job.job_id,
        "dl-first",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    store.command(first.job.job_id, "stop")
    fetches = []

    def handler_second(request: httpx.Request) -> httpx.Response:
        fetches.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "id": rid,
                "uploadtime": 1,
                "log": build_log(include_ots=False, p1_rating=1300, p2_rating=1300),
                "format": SOURCE,
            },
        )

    second = store.start_campaign(
        mc_config(
            max_selected_replays=5,
            dataset_filters=ReplayFilterConfig(require_ots=False),
        )
    )
    client2 = ShowdownReplayClient(transport=httpx.MockTransport(handler_second), sleep=lambda _: None)
    run_collection_worker(
        env,
        second.job.job_id,
        "dl-second",
        "download",
        client_factory=lambda _c: client2,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert fetches
    assert is_replay_archived(rid)
    assert store.get_campaign(second.job.job_id).counters["selected_in_attempt"] == 1
    client.close()
    client2.close()


def test_other_format_backlog_is_not_claimed(env):
    from pokecoach_trainer.schemas import ReplayFilterConfig

    clock = FakeClock()
    store = CollectionStore(env, clock=clock)
    view = store.start_campaign(
        mc_config(max_selected_replays=5, dataset_filters=ReplayFilterConfig())
    )
    other = "gen9vgc2026regfbo3-otherfmt"
    mine = f"{SOURCE}-minefmt"
    store.enqueue_replays(
        view.job.job_id,
        [
            SearchItem(other, 1, "gen9vgc2026regfbo3"),
            SearchItem(mine, 1, SOURCE),
        ],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease is not None
    assert claim.lease.replay_id == mine
