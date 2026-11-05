from datetime import datetime, timezone
from dataclasses import replace

import pytest

from pokecoach_trainer.archive_catalog import catalog_connection, catalog_versions
from pokecoach_trainer.archive_service import archive_replay, is_replay_archived
from pokecoach_trainer.collection_publication import (
    clear_publication_hooks,
    prepare_replay,
    publish_replay,
    recover_publications,
    set_publication_hook,
)
from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.errors import StaleLease
from pokecoach_trainer.schemas import ArchiveSource
from pokecoach_trainer.settings import clear_settings_cache


@pytest.fixture
def pub_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    yield tmp_path
    clear_publication_hooks()


def _payload(replay_id: str, *, log: str = "|turn|1|\n") -> dict:
    return {
        "id": replay_id,
        "uploadtime": 1_700_000_000,
        "log": log,
        "format": replay_id.split("-")[0],
    }


def test_same_id_hash_publishes_once_under_race(pub_env):
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-race"
    ref1 = archive_replay(replay_id, _payload(replay_id), source)
    ref2 = archive_replay(replay_id, _payload(replay_id), source)
    assert ref1.content_sha256 == ref2.content_sha256
    assert is_replay_archived(replay_id)


def test_recover_after_temp_fsync(pub_env):
    store = CollectionStore(pub_env)
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-rec"
    prepared = prepare_replay(pub_env, replay_id, _payload(replay_id), source)
    publish_replay(store, prepared, lease=None)
    assert recover_publications(store) >= 0


def test_different_hash_keeps_both_versions(pub_env):
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-ver"
    first = archive_replay(replay_id, _payload(replay_id, log="|turn|1|\n"), source)
    second = archive_replay(replay_id, _payload(replay_id, log="|turn|2|\n"), source)
    versions = catalog_versions(pub_env, replay_id)
    assert len(versions) == 2
    assert first.content_sha256 != second.content_sha256


def test_import_racing_with_download_has_one_first_publication(pub_env):
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-race2"
    payload = _payload(replay_id)
    ref_a = archive_replay(replay_id, payload, source)
    store = CollectionStore(pub_env)
    prepared = prepare_replay(pub_env, replay_id, payload, source)
    ref_b = publish_replay(store, prepared, lease=None)
    assert ref_a.content_sha256 == ref_b.ref.content_sha256
    assert len(catalog_versions(pub_env, replay_id)) == 1


def test_recover_after_rename_before_sql_commit(pub_env):
    store = CollectionStore(pub_env)
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-hook"
    prepared = prepare_replay(pub_env, replay_id, _payload(replay_id), source)

    def crash() -> None:
        raise RuntimeError("simulated crash")

    set_publication_hook("after_rename_before_catalog", crash)
    try:
        publish_replay(store, prepared, lease=None)
    except RuntimeError:
        pass
    recovered = recover_publications(store)
    assert recovered >= 1
    assert is_replay_archived(replay_id)


def test_disk_full_does_not_mark_archived(pub_env, monkeypatch):
    import os

    store = CollectionStore(pub_env)
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-full"
    prepared = prepare_replay(pub_env, replay_id, _payload(replay_id), source)

    def boom(a, b):  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        publish_replay(store, prepared, lease=None)
    assert not is_replay_archived(replay_id)


def test_reassigned_worker_cannot_publish(pub_env):
    from pokecoach_trainer.collection_types import SearchItem
    from pokecoach_trainer.schemas import CollectionConfig

    store = CollectionStore(pub_env)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=5))
    replay_id = "gen9championsvgc2026regmc-stalepub"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id, 1, "gen9championsvgc2026regmc")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    store.release_lease(claim.lease)
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    prepared = prepare_replay(pub_env, replay_id, _payload(replay_id), source)
    with pytest.raises(StaleLease):
        publish_replay(store, prepared, lease=claim.lease)


def test_missing_final_object_is_not_success(pub_env):
    store = CollectionStore(pub_env)
    with store._tx() as conn:
        conn.execute(
            """
            INSERT INTO publication_intents(
                intent_id, replay_id, content_sha256, attempt_id, lease_token,
                temp_relpath, final_relpath, metadata_json, phase
            ) VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, 'prepared')
            """,
            (
                "intent-missing",
                "gen9championsvgc2026regmc-missing",
                "deadbeef",
                "staging/missing.tmp",
                "objects/gen9championsvgc2026regmc-missing/deadbeef.json",
                "{}",
            ),
        )
    assert recover_publications(store) == 0
    assert not is_replay_archived("gen9championsvgc2026regmc-missing")


def test_rejected_payload_never_reaches_publication_or_logs(pub_env):
    from pokecoach_trainer.collection_filters import evaluate_selection
    from pokecoach_trainer.collection_types import SearchItem
    from pokecoach_trainer.schemas import CollectionConfig, TeamFilter

    team = TeamFilter(teams=[["pikachu", "eevee", "bulbasaur", "charmander", "squirtle", "mew"]])
    store = CollectionStore(pub_env)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=5, team_filter=team)
    )
    replay_id = "gen9championsvgc2026regmc-reject"
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id, 1, "gen9championsvgc2026regmc")],
    )
    claim = store.claim_download(view.job.job_id, "w1")
    assert claim.lease
    decision = evaluate_selection(
        CollectionConfig(format_ids=["mc"], team_filter=team),
        uploadtime=1,
        summary=None,
    )
    store.record_selection(claim.lease, decision, None)
    with catalog_connection(pub_env) as conn:
        intents = conn.execute("SELECT COUNT(*) AS c FROM publication_intents").fetchone()["c"]
    staging = list((pub_env / "archive" / "staging").glob("*")) if (pub_env / "archive" / "staging").is_dir() else []
    assert intents == 0
    assert staging == []
