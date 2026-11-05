from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pokecoach_trainer.archive_service import archive_replay, create_snapshot
from pokecoach_trainer.dataset_service import create_dataset
from pokecoach_trainer.main import app
from pokecoach_trainer.schemas import (
    ArchiveSource,
    DatasetJobConfig,
    DatasetRecipe,
    JobKind,
    JobRecord,
    JobStatus,
    PrepareExecutionConfig,
    SnapshotDatasetSource,
)
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout
from replay_log_fixtures import build_log


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield TestClient(app)


def _snapshot_id(tmp_path) -> str:
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-api"
    archive_replay(
        replay_id,
        {
            "id": replay_id,
            "uploadtime": 1,
            "log": build_log(),
            "format": "gen9championsvgc2026regmc",
        },
        source,
    )
    snap = create_snapshot(__import__("pokecoach_trainer.schemas", fromlist=["SnapshotRequest"]).SnapshotRequest(format_ids=["gen9championsvgc2026regmc"]))
    return snap.snapshot_id


def test_create_and_prepare_returns_202(api_env, tmp_path):
    snap_id = _snapshot_id(tmp_path)
    recipe = DatasetRecipe(target_replays=1)
    created = api_env.post(
        "/datasets",
        json={"snapshot_id": snap_id, "recipe": recipe.model_dump(mode="json")},
    )
    assert created.status_code == 200
    dataset_id = created.json()["dataset_id"]
    with patch("pokecoach_trainer.api.datasets.submit_preparation") as mock_submit:
        mock_submit.return_value = JobRecord(
            job_id="job-prep-1",
            kind=JobKind.prepare,
            status=JobStatus.running,
            config=DatasetJobConfig(
                dataset_id=dataset_id,
                dataset_source=SnapshotDatasetSource(snapshot_id=snap_id),
            ),
        )
        resp = api_env.post(f"/datasets/{dataset_id}/prepare")
    assert resp.status_code == 202
    mock_submit.assert_called_once()
    assert mock_submit.call_args.args[0] == dataset_id
    execution = mock_submit.call_args.kwargs.get("execution")
    assert execution is not None
    assert execution.conversion_workers == "auto"


def test_ready_dataset_prepare_conflict(api_env, tmp_path):
    snap_id = _snapshot_id(tmp_path)
    recipe = DatasetRecipe(target_replays=1)
    manifest = create_dataset(snap_id, recipe, dataset_id="ds-ready")
    from pokecoach_trainer.dataset_service import get_dataset
    from pokecoach_trainer.schemas import DatasetState
    from pokecoach_trainer.storage import write_json_atomic

    manifest.state = DatasetState.ready
    write_json_atomic(
        __import__("pokecoach_trainer.settings", fromlist=["storage_path"]).storage_path(
            "datasets", "ds-ready", "manifest.json"
        ),
        manifest.model_dump(mode="json"),
    )
    resp = api_env.post("/datasets/ds-ready/prepare")
    assert resp.status_code == 409


def test_snapshot_api_forwards_execution_to_job(api_env, tmp_path):
    snap_id = _snapshot_id(tmp_path)
    recipe = DatasetRecipe(target_replays=1)
    created = api_env.post(
        "/datasets",
        json={"snapshot_id": snap_id, "recipe": recipe.model_dump(mode="json")},
    )
    dataset_id = created.json()["dataset_id"]
    with patch("pokecoach_trainer.api.datasets.submit_preparation") as mock_submit:
        mock_submit.return_value = JobRecord(
            job_id="job-prep-exec",
            kind=JobKind.prepare,
            status=JobStatus.running,
            config=DatasetJobConfig(
                dataset_id=dataset_id,
                dataset_source=SnapshotDatasetSource(snapshot_id=snap_id),
                execution=PrepareExecutionConfig(conversion_workers=4),
            ),
        )
        resp = api_env.post(
            f"/datasets/{dataset_id}/prepare",
            json={"conversion_workers": 4},
        )
    assert resp.status_code == 202
    mock_submit.assert_called_once()
    execution = mock_submit.call_args.kwargs.get("execution")
    assert execution is not None
    assert execution.conversion_workers == 4


def test_snapshot_prepare_body_absent_defaults_auto(api_env, tmp_path):
    snap_id = _snapshot_id(tmp_path)
    recipe = DatasetRecipe(target_replays=1)
    created = api_env.post(
        "/datasets",
        json={"snapshot_id": snap_id, "recipe": recipe.model_dump(mode="json")},
    )
    dataset_id = created.json()["dataset_id"]
    with patch("pokecoach_trainer.api.datasets.submit_preparation") as mock_submit:
        mock_submit.return_value = JobRecord(
            job_id="job-auto",
            kind=JobKind.prepare,
            status=JobStatus.running,
            config=DatasetJobConfig(
                dataset_id=dataset_id,
                dataset_source=SnapshotDatasetSource(snapshot_id=snap_id),
            ),
        )
        resp = api_env.post(f"/datasets/{dataset_id}/prepare")
    assert resp.status_code == 202
    execution = mock_submit.call_args.kwargs.get("execution")
    assert execution is not None
    assert execution.conversion_workers == "auto"
