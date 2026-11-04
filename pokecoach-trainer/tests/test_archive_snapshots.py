from datetime import datetime, timezone

import pytest

from pokecoach_trainer.archive_service import (
    archive_replay,
    create_snapshot,
    load_index_entries,
    read_snapshot,
)
from pokecoach_trainer.errors import Conflict
from pokecoach_trainer.schemas import ArchiveSource, SnapshotRequest
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout


@pytest.fixture
def archive_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield


def _archive(replay_id: str, log: str, uploadtime: int) -> None:
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    archive_replay(
        replay_id,
        {
            "id": replay_id,
            "uploadtime": uploadtime,
            "log": log,
            "format": "gen9championsvgc2026regmc",
        },
        source,
    )


def test_snapshot_stable_hash(archive_env):
    _archive("gen9championsvgc2026regmc-a1", "|turn|1\n", 1_700_000_000)
    request = SnapshotRequest(format_ids=["mc"])
    snap1 = create_snapshot(request)
    snap2 = create_snapshot(request)
    assert snap1.snapshot_sha256 == snap2.snapshot_sha256
    assert read_snapshot(snap1.snapshot_id).snapshot_sha256 == snap1.snapshot_sha256


def test_snapshot_version_conflict_requires_selection(archive_env):
    replay_id = "gen9championsvgc2026regmc-conflict"
    _archive(replay_id, "|turn|1\n", 1)
    _archive(replay_id, "|turn|2\n", 2)
    with pytest.raises(Conflict):
        create_snapshot(SnapshotRequest(format_ids=["mc"]))
    versions = [row for row in load_index_entries() if row["replay_id"] == replay_id]
    assert len(versions) == 2
    selected = create_snapshot(
        SnapshotRequest(
            format_ids=["mc"],
            replay_version_selection={replay_id: versions[0]["content_sha256"]},
        )
    )
    assert len(selected.refs) == 1
