import json
from datetime import datetime, timezone

import pytest

from pokecoach_trainer.archive_service import (
    archive_replay,
    import_legacy_logs,
    is_replay_archived,
    load_index_entries,
    read_archived_object,
    rebuild_index_from_objects,
)
from pokecoach_trainer.errors import InvalidConfiguration
from pokecoach_trainer.schemas import ArchiveSource
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout, storage_path


@pytest.fixture
def archive_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield tmp_path


def _payload(replay_id: str, log: str, uploadtime: int = 1_700_000_000) -> dict:
    return {
        "id": replay_id,
        "uploadtime": uploadtime,
        "log": log,
        "format": replay_id.split("-")[0],
    }


def test_archive_low_elo_without_ots_preserved(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    log = "|player|p1|LowElo|1000\n|player|p2|Other|1000\n|turn|1\n"
    replay_id = "gen9championsvgc2026regmc-abc"
    ref = archive_replay(replay_id, _payload(replay_id, log), source)
    envelope = read_archived_object(replay_id, ref.content_sha256)
    assert envelope["provider"]["log"] == log
    assert is_replay_archived(replay_id)


def test_archive_deduplicates_same_content(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-dup"
    log = "|turn|1\n"
    ref1 = archive_replay(replay_id, _payload(replay_id, log), source)
    ref2 = archive_replay(replay_id, _payload(replay_id, log), source)
    assert ref1.content_sha256 == ref2.content_sha256
    assert len(load_index_entries()) == 1


def test_archive_rejects_regf_replay(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9vgc2026regfbo3-notallowed"
    with pytest.raises(InvalidConfiguration):
        archive_replay(replay_id, _payload(replay_id, "|turn|1\n"), source)


def test_archive_rejects_id_mismatch(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    with pytest.raises(InvalidConfiguration):
        archive_replay("gen9championsvgc2026regmc-x", _payload("gen9championsvgc2026regmc-y", "|turn|1\n"), source)


def test_rebuild_index_from_objects(archive_env):
    from pokecoach_trainer.archive_catalog import catalog_has_replay

    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-rebuild"
    archive_replay(replay_id, _payload(replay_id, "|turn|1\n"), source)
    index_path = storage_path("archive", "index", "replays.jsonl")
    if index_path.is_file():
        index_path.unlink()
    count = rebuild_index_from_objects()
    assert count == 1
    assert catalog_has_replay(archive_env, replay_id)


def test_import_legacy_logs_deduplicates(archive_env):
    inbox = storage_path("inbox", "legacy.json")
    replay_id = "gen9championsvgc2026regmc-legacy1"
    payload = {replay_id: [1_700_000_000, "|turn|1\n"]}
    inbox.write_text(json.dumps(payload), encoding="utf-8")
    snap1 = import_legacy_logs("legacy.json")
    snap2 = import_legacy_logs("legacy.json")
    assert snap1.snapshot_sha256 == snap2.snapshot_sha256
    assert len(load_index_entries()) == 1


def test_import_legacy_rejects_duplicate_keys(archive_env):
    inbox = storage_path("inbox", "bad.json")
    inbox.write_text('{"a": [1, "x"], "a": [2, "y"]}', encoding="utf-8")
    with pytest.raises(InvalidConfiguration):
        import_legacy_logs("bad.json")
