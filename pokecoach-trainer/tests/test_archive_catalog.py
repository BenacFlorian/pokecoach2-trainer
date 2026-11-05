"""Tests catalogue SQLite de l'archive."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from datetime import datetime, timezone

import pytest

from pokecoach_trainer.archive_catalog import (
    catalog_connection,
    catalog_has_replay,
    catalog_versions,
    check_sqlite_version,
    initialize_catalog,
    insert_replay_version,
)
from pokecoach_trainer.errors import StorageIncompatible
from pokecoach_trainer.schemas import ArchiveRef, ArchiveSource
from pokecoach_trainer.settings import clear_settings_cache


@pytest.fixture
def catalog_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "dummy.txt").write_text("keep", encoding="utf-8")
    yield tmp_path


def test_empty_init_is_idempotent_and_preserves_other_roots(catalog_root):
    initialize_catalog(catalog_root)
    initialize_catalog(catalog_root)
    assert catalog_has_replay(catalog_root, "missing-id") is False
    assert (catalog_root / "models" / "dummy.txt").read_text(encoding="utf-8") == "keep"


def test_legacy_state_is_not_imported(catalog_root):
    objects = catalog_root / "archive" / "objects" / "gen9championsvgc2026regmc-x"
    objects.mkdir(parents=True)
    (objects / "abc.json").write_text("{}", encoding="utf-8")
    (catalog_root / "archive" / "index").mkdir(parents=True)
    (catalog_root / "archive" / "index" / "replays.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(StorageIncompatible):
        initialize_catalog(catalog_root)


def test_duplicate_version_is_rejected(catalog_root):
    initialize_catalog(catalog_root)
    ref = ArchiveRef(
        replay_id="gen9championsvgc2026regmc-dup",
        content_sha256="a" * 64,
        log_sha256="b" * 64,
        format_id="gen9championsvgc2026regmc",
        uploadtime=1,
        object_relpath="objects/x/a.json",
    )
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    insert_replay_version(
        catalog_root,
        ref=ref,
        source=source,
        provider_format="gen9championsvgc2026regmc",
    )
    with pytest.raises(sqlite3.IntegrityError):
        insert_replay_version(
            catalog_root,
            ref=ref,
            source=source,
            provider_format="gen9championsvgc2026regmc",
        )


def test_catalog_lookup_is_indexed(catalog_root):
    initialize_catalog(catalog_root)
    replay_id = "gen9championsvgc2026regmc-idx"
    ref = ArchiveRef(
        replay_id=replay_id,
        content_sha256="c" * 64,
        log_sha256="d" * 64,
        format_id="gen9championsvgc2026regmc",
        uploadtime=2,
        object_relpath="objects/x/c.json",
    )
    insert_replay_version(
        catalog_root,
        ref=ref,
        source=ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc)),
        provider_format="gen9championsvgc2026regmc",
    )
    with catalog_connection(catalog_root) as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM replay_versions WHERE replay_id = ?",
            (replay_id,),
        ).fetchall()
    plan_text = " ".join(str(row[3]) for row in plan)
    assert "USING INDEX" in plan_text or "USING COVERING INDEX" in plan_text
    assert catalog_has_replay(catalog_root, replay_id)
    assert catalog_versions(catalog_root, replay_id)[0].content_sha256 == ref.content_sha256


def test_unfixed_sqlite_is_rejected():
    with pytest.raises(StorageIncompatible):
        check_sqlite_version("3.40.0")
