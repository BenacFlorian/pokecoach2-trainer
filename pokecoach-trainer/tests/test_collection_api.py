from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pokecoach_trainer.archive_service import archive_replay
from pokecoach_trainer.main import app
from pokecoach_trainer.schemas import (
    ArchiveSource,
    CollectionConfig,
    CollectionJobConfig,
    ExperimentConfig,
    JobKind,
    JobRecord,
    JobStatus,
)
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield TestClient(app)


def test_collection_job_wrong_kind_returns_404(api_env, tmp_path):
    from pokecoach_trainer.jobs import save_job

    record = JobRecord(
        job_id="job-training-x",
        kind=JobKind.training,
        status=JobStatus.completed,
        config=ExperimentConfig(name="x", dataset_id="ds1"),
    )
    save_job(record)
    resp = api_env.get("/collection/jobs/job-training-x")
    assert resp.status_code == 404


@patch("pokecoach_trainer.api.collection.submit_collection")
def test_collection_post_returns_202(mock_submit, api_env):
    mock_submit.return_value = JobRecord(
        job_id="job-col-1",
        kind=JobKind.collection,
        status=JobStatus.running,
        config=CollectionJobConfig(config=CollectionConfig(format_ids=["mc"])),
    )
    resp = api_env.post(
        "/collection/jobs",
        json={"kind": "collection", "config": {"format_ids": ["mc"]}},
    )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job-col-1"
    mock_submit.assert_called_once()


def test_archive_list_and_snapshot(api_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-api1"
    archive_replay(
        replay_id,
        {
            "id": replay_id,
            "uploadtime": 1_700_000_000,
            "log": "|turn|1\n",
            "format": "gen9championsvgc2026regmc",
        },
        source,
    )
    listing = api_env.get("/archive/replays")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    snap = api_env.post(
        "/archive/snapshots",
        json={"format_ids": ["gen9championsvgc2026regmc"]},
    )
    assert snap.status_code == 200
    snapshot_id = snap.json()["snapshot_id"]
    detail = api_env.get(f"/archive/snapshots/{snapshot_id}")
    assert detail.status_code == 200
