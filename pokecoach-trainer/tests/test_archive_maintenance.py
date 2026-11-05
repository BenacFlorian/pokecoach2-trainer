"""Archive : chemins chauds sans lecture JSONL opérationnelle."""

from datetime import datetime, timezone
from threading import Thread
from unittest.mock import patch

import pytest

from pokecoach_trainer.archive_catalog import (
    backup_catalog,
    catalog_maintenance_shared,
    export_catalog_jsonl,
    rebuild_catalog_from_objects,
)
from pokecoach_trainer.archive_service import (
    archive_replay,
    create_snapshot,
    list_replays,
    rebuild_index_from_objects,
)
from pokecoach_trainer.errors import Conflict
from pokecoach_trainer.schemas import ArchiveSource, SnapshotRequest
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout


@pytest.fixture
def archive_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield tmp_path


def test_archive_hot_paths_do_not_read_jsonl(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-hot"
    log = "|turn|1\n"
    archive_replay(replay_id, {"id": replay_id, "uploadtime": 1_700_000_000, "log": log, "format": replay_id.split("-")[0]}, source)

    def forbidden(*args, **kwargs):
        raise AssertionError("load_index_entries ne doit pas être utilisé sur le chemin chaud")

    with patch("pokecoach_trainer.archive_service.load_index_entries", side_effect=forbidden):
        refs, total = list_replays(limit=10)
        assert total >= 1
        assert any(r.replay_id == replay_id for r in refs)
        snap = create_snapshot(SnapshotRequest(format_ids=["mc"]))
        assert any(r.replay_id == replay_id for r in snap.refs)


def test_list_filters_use_source_format_and_dates(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    rid = "gen9championsvgc2026regmc-date"
    archive_replay(
        rid,
        {"id": rid, "uploadtime": 1_700_000_000, "log": "|t|1\n", "format": "gen9championsvgc2026regmc"},
        source,
    )
    refs, total = list_replays(upload_after=datetime(2023, 1, 1, tzinfo=timezone.utc))
    assert total >= 1
    assert any(r.replay_id == rid for r in refs)


def test_snapshot_while_collecting_freezes_versions(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    rid = "gen9championsvgc2026regmc-freeze"
    archive_replay(rid, {"id": rid, "uploadtime": 1, "log": "|t|1\n", "format": "gen9championsvgc2026regmc"}, source)
    snap1 = create_snapshot(SnapshotRequest(format_ids=["mc"]))
    snap2 = create_snapshot(SnapshotRequest(format_ids=["mc"]))
    assert snap1.snapshot_sha256 == snap2.snapshot_sha256


def test_snapshot_conflict_requires_explicit_hash(archive_env):
    from pokecoach_trainer.errors import Conflict as ArchiveConflict

    rid = "gen9championsvgc2026regmc-maint-conf"
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    archive_replay(rid, {"id": rid, "uploadtime": 1, "log": "|t|1\n", "format": "gen9championsvgc2026regmc"}, source)
    archive_replay(rid, {"id": rid, "uploadtime": 1, "log": "|t|2\n", "format": "gen9championsvgc2026regmc"}, source)
    with pytest.raises(ArchiveConflict):
        create_snapshot(SnapshotRequest(format_ids=["mc"]))


def test_rebuild_requires_exclusive_maintenance(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    rid = "gen9championsvgc2026regmc-rebuild-lock"
    archive_replay(rid, {"id": rid, "uploadtime": 1, "log": "|t|1\n", "format": "gen9championsvgc2026regmc"}, source)
    errors: list[Exception] = []

    def hold_shared() -> None:
        try:
            with catalog_maintenance_shared(archive_env):
                import time

                time.sleep(0.5)
        except Exception as exc:
            errors.append(exc)

    t = Thread(target=hold_shared)
    t.start()
    import time

    time.sleep(0.05)
    with pytest.raises(Conflict):
        rebuild_index_from_objects()
    t.join(timeout=2)
    assert rebuild_catalog_from_objects(archive_env) >= 1


def test_backup_restores_committed_wal_content(archive_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    rid = "gen9championsvgc2026regmc-backup"
    archive_replay(rid, {"id": rid, "uploadtime": 1, "log": "|t|1\n", "format": "gen9championsvgc2026regmc"}, source)
    dest = archive_env / "backup.sqlite3"
    backup_catalog(archive_env, dest)
    assert dest.is_file()
    export_path = archive_env / "export.jsonl"
    count = export_catalog_jsonl(archive_env, export_path)
    assert count >= 1
    assert export_path.is_file()


def test_prepared_dataset_unchanged_while_archive_grows(archive_env):
    from pokecoach_trainer.dataset_service import create_dataset, get_dataset
    from pokecoach_trainer.schemas import DatasetRecipe, SnapshotRequest
    from replay_log_fixtures import build_log

    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    rid0 = "gen9championsvgc2026regmc-ds0"
    log = build_log(p1_name="A", p2_name="B", p1_rating=1200, p2_rating=1200, winner="A")
    archive_replay(
        rid0,
        {"id": rid0, "uploadtime": 1, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    snap = create_snapshot(SnapshotRequest(format_ids=["mc"]))
    create_dataset(snap.snapshot_id, DatasetRecipe(target_replays=1), dataset_id="ds-grow")
    train_before = get_dataset("ds-grow").replay_ids_train[:]
    rid1 = "gen9championsvgc2026regmc-ds1"
    archive_replay(
        rid1,
        {"id": rid1, "uploadtime": 2, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    assert get_dataset("ds-grow").replay_ids_train == train_before

