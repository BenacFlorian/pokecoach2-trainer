import sqlite3

from pokecoach_trainer.archive_catalog import (
    CATALOG_VERSION,
    _SCHEMA_V1,
    catalog_connection,
    catalog_path,
    initialize_catalog,
)
from pokecoach_trainer.errors import StorageIncompatible
from pokecoach_trainer.settings import clear_settings_cache


def test_migrate_v1_preserves_rows_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    (tmp_path / "archive").mkdir()
    path = catalog_path(tmp_path)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_V1)
    conn.execute("PRAGMA user_version=1")
    conn.execute(
        """
        INSERT INTO campaigns(
            campaign_id, request_fingerprint, request_config_json, effective_config_json,
            phase, attempt_id, discovery_workers, download_workers, created_at, updated_at
        ) VALUES ('job-old', 'fp', '{}', '{}', 'stopped', 'attempt-1', 1, 1, 1, 1)
        """
    )
    conn.commit()
    conn.close()
    initialize_catalog(tmp_path)
    initialize_catalog(tmp_path)
    with catalog_connection(tmp_path) as migrated:
        version = int(migrated.execute("PRAGMA user_version").fetchone()[0])
        assert version == CATALOG_VERSION
        row = migrated.execute("SELECT campaign_id FROM campaigns").fetchone()
        assert row["campaign_id"] == "job-old"
        cols = {r[1] for r in migrated.execute("PRAGMA table_info(campaigns)").fetchall()}
        assert "validation_parser_version" in cols
        migrated.execute("SELECT 1 FROM replay_metadata_cache")
        migrated.execute("SELECT 1 FROM campaign_log_claims")


def test_unknown_schema_version_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    (tmp_path / "archive").mkdir()
    path = catalog_path(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version=99")
    conn.commit()
    conn.close()
    try:
        initialize_catalog(tmp_path)
        raised = False
    except StorageIncompatible:
        raised = True
    assert raised
