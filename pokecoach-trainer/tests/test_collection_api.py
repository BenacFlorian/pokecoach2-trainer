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


@patch("pokecoach_trainer.api.collection.submit_collection")
def test_post_returns_202_without_waiting_for_network(mock_submit, api_env):
    mock_submit.return_value = JobRecord(
        job_id="job-col-fast",
        kind=JobKind.collection,
        status=JobStatus.queued,
        config=CollectionJobConfig(config=CollectionConfig(format_ids=["mc"])),
    )
    resp = api_env.post(
        "/collection/jobs",
        json={"kind": "collection", "config": {"format_ids": ["mc"]}},
    )
    assert resp.status_code == 202
    mock_submit.assert_called_once()


@patch("pokecoach_trainer.api.collection.submit_collection")
def test_double_post_returns_same_job(mock_submit, api_env):
    record = JobRecord(
        job_id="job-same",
        kind=JobKind.collection,
        status=JobStatus.running,
        config=CollectionJobConfig(config=CollectionConfig(format_ids=["mc"])),
    )
    mock_submit.return_value = record
    r1 = api_env.post("/collection/jobs", json={"kind": "collection", "config": {"format_ids": ["mc"]}})
    r2 = api_env.post("/collection/jobs", json={"kind": "collection", "config": {"format_ids": ["mc"]}})
    assert r1.json()["job_id"] == r2.json()["job_id"]


@patch("pokecoach_trainer.api.collection.submit_collection")
def test_different_active_post_returns_409(mock_submit, api_env):
    from pokecoach_trainer.errors import Conflict

    mock_submit.side_effect = [
        JobRecord(
            job_id="job-a",
            kind=JobKind.collection,
            status=JobStatus.running,
            config=CollectionJobConfig(config=CollectionConfig(format_ids=["mc"])),
        ),
        Conflict("autre campagne active"),
    ]
    api_env.post("/collection/jobs", json={"kind": "collection", "config": {"format_ids": ["mc"]}})
    resp = api_env.post(
        "/collection/jobs",
        json={"kind": "collection", "config": {"format_ids": ["mc"], "mode": "incremental"}},
    )
    assert resp.status_code == 409


def test_controls_wrong_kind_return_404(api_env, tmp_path):
    from pokecoach_trainer.jobs import save_job

    save_job(
        JobRecord(
            job_id="job-train-1",
            kind=JobKind.training,
            status=JobStatus.completed,
            config=ExperimentConfig(name="x", dataset_id="ds1"),
        )
    )
    assert api_env.post("/collection/jobs/job-train-1/pause").status_code == 404


def test_resume_after_budget_requires_new_attempt(api_env):
    from pokecoach_trainer.collection_store import CollectionStore, SystemClock
    from pokecoach_trainer.archive_catalog import catalog_connection
    from pokecoach_trainer.settings import get_settings

    store = CollectionStore(get_settings().storage_root, clock=SystemClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=1))
    with catalog_connection(get_settings().storage_root) as conn:
        conn.execute(
            "UPDATE campaigns SET phase='budget_reached' WHERE campaign_id=?",
            (view.job.job_id,),
        )
    resp = api_env.post(f"/collection/jobs/{view.job.job_id}/resume", json={})
    assert resp.status_code == 409


def test_unknown_field_is_422(api_env):
    resp = api_env.post(
        "/collection/jobs",
        json={"kind": "collection", "config": {"format_ids": ["mc"], "unknown_field": True}},
    )
    assert resp.status_code == 422


def test_campaign_replays_and_snapshot_routes(api_env):
    from datetime import datetime, timezone

    from pokecoach_trainer.archive_service import archive_replay
    from pokecoach_trainer.collection_store import CollectionStore, SystemClock
    from pokecoach_trainer.schemas import ArchiveSource
    from pokecoach_trainer.settings import get_settings

    root = get_settings().storage_root
    store = CollectionStore(root, clock=SystemClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    rid = "gen9championsvgc2026regmc-api-sel"
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    archive_replay(
        rid,
        {"id": rid, "uploadtime": 1, "log": "|t|1\n", "format": "gen9championsvgc2026regmc"},
        source,
    )
    with store._tx() as conn:
        conn.execute(
            """
            INSERT INTO campaign_replays(campaign_id, replay_id, decision, reason, filter_sha256)
            VALUES (?, ?, 'selected', 'selected', 'f')
            ON CONFLICT(campaign_id, replay_id) DO UPDATE SET decision='selected'
            """,
            (view.job.job_id, rid),
        )
    listing = api_env.get(f"/collection/jobs/{view.job.job_id}/replays")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    snap = api_env.post(f"/collection/jobs/{view.job.job_id}/snapshot")
    assert snap.status_code == 200


def test_campaign_listing_and_snapshot_exclude_other_campaign_objects(api_env):
    from datetime import datetime, timezone

    from pokecoach_trainer.archive_service import archive_replay
    from pokecoach_trainer.collection_store import CollectionStore, SystemClock
    from pokecoach_trainer.schemas import ArchiveSource
    from pokecoach_trainer.settings import get_settings

    root = get_settings().storage_root
    store = CollectionStore(root, clock=SystemClock())
    view_a = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    store.command(view_a.job.job_id, "stop")
    view_b = store.start_campaign(CollectionConfig(format_ids=["mc"], mode="incremental"))
    rid_a = "gen9championsvgc2026regmc-camp-a"
    rid_b = "gen9championsvgc2026regmc-camp-b"
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    for rid in (rid_a, rid_b):
        archive_replay(
            rid,
            {"id": rid, "uploadtime": 1, "log": "|t|1\n", "format": "gen9championsvgc2026regmc"},
            source,
        )
    with store._tx() as conn:
        conn.execute(
            """
            INSERT INTO campaign_replays(campaign_id, replay_id, decision, reason, filter_sha256)
            VALUES (?, ?, 'selected', 'selected', 'fa')
            """,
            (view_a.job.job_id, rid_a),
        )
        conn.execute(
            """
            INSERT INTO campaign_replays(campaign_id, replay_id, decision, reason, filter_sha256)
            VALUES (?, ?, 'selected', 'selected', 'fb')
            """,
            (view_b.job.job_id, rid_b),
        )
    listing_b = api_env.get(f"/collection/jobs/{view_b.job.job_id}/replays")
    assert listing_b.status_code == 200
    ids_b = {r["replay_id"] for r in listing_b.json()["refs"]}
    assert rid_b in ids_b
    assert rid_a not in ids_b
    snap_b = api_env.post(f"/collection/jobs/{view_b.job.job_id}/snapshot")
    assert snap_b.status_code == 200
    assert all(ref["replay_id"] != rid_a for ref in snap_b.json()["refs"])


def test_legacy_json_collection_resume_returns_error(api_env):
    from pokecoach_trainer.jobs import save_job
    from pokecoach_trainer.schemas import CollectionJobConfig

    save_job(
        JobRecord(
            job_id="job-legacy-json",
            kind=JobKind.collection,
            status=JobStatus.interrupted,
            config=CollectionJobConfig(config=CollectionConfig(format_ids=["mc"])),
        )
    )
    resp = api_env.post("/collection/jobs/job-legacy-json/resume", json={})
    assert resp.status_code == 422
    assert "non reprenable" in resp.json()["detail"]["message"]

