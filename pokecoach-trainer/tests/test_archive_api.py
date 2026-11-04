from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from pokecoach_trainer.archive_service import archive_replay
from pokecoach_trainer.main import app
from pokecoach_trainer.schemas import ArchiveSource
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield TestClient(app)


def test_snapshot_not_found(api_env):
    resp = api_env.get("/archive/snapshots/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_archive_pagination(api_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    for idx in range(3):
        replay_id = f"gen9championsvgc2026regmc-p{idx}"
        archive_replay(
            replay_id,
            {
                "id": replay_id,
                "uploadtime": 1_700_000_000 + idx,
                "log": f"|turn|{idx}\n",
                "format": "gen9championsvgc2026regmc",
            },
            source,
        )
    resp = api_env.get("/archive/replays", params={"limit": 2, "offset": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
