from datetime import datetime, timezone

import pytest

from pokecoach_trainer.archive_service import archive_replay
from pokecoach_trainer.replay_metadata import extract_replay_metadata
from pokecoach_trainer.schemas import ArchiveRef, ArchiveSource
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout
from replay_log_fixtures import build_log


@pytest.fixture
def archive_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield


def _ref(replay_id: str, content_sha256: str = "abc") -> ArchiveRef:
    return ArchiveRef(
        replay_id=replay_id,
        content_sha256=content_sha256,
        log_sha256="loghash",
        format_id="gen9championsvgc2026regmc",
        object_relpath=f"objects/{replay_id}/{content_sha256}.json",
    )


def test_ratings_1300_1000_p1_only_perspective(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-ratings"
    log = build_log(p1_rating=1300, p2_rating=1000)
    ref = archive_replay(
        replay_id,
        {"id": replay_id, "uploadtime": 1, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    meta = extract_replay_metadata(ref)
    assert meta.p1_rating == 1300
    assert meta.p2_rating == 1000


def test_rating_1100_inclusive(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-1100"
    log = build_log(p1_rating=1100, p2_rating=900)
    ref = archive_replay(
        replay_id,
        {"id": replay_id, "uploadtime": 1, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    meta = extract_replay_metadata(ref)
    assert meta.p1_rating == 1100


def test_forfeit_three_turns_accepted(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-forfeit3"
    log = build_log(turns=3, termination="forfeit")
    ref = archive_replay(
        replay_id,
        {"id": replay_id, "uploadtime": 1, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    meta = extract_replay_metadata(ref)
    assert meta.termination == "forfeit"
    assert meta.completed_turns == 3


def test_forfeit_two_turns_count(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-forfeit2"
    log = build_log(turns=2, termination="forfeit")
    ref = archive_replay(
        replay_id,
        {"id": replay_id, "uploadtime": 1, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    meta = extract_replay_metadata(ref)
    assert meta.completed_turns == 2


def test_turn_three_without_resolution_unknown(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-turn3open"
    lines = build_log(turns=2).splitlines()
    lines.append("|turn|3|")
    lines.append("|raw|Bob forfeited.")
    log = "\n".join(lines) + "\n"
    ref = archive_replay(
        replay_id,
        {"id": replay_id, "uploadtime": 1, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    meta = extract_replay_metadata(ref)
    assert meta.completed_turns == 2
