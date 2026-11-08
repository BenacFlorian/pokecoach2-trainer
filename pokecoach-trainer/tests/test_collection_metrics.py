"""Tests métriques collecte."""

from __future__ import annotations

from pokecoach_trainer.collection_metrics import collection_progress
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.job_service import read_collection_worker_logs, read_job_logs
from pokecoach_trainer.schemas import CollectionConfig
from pokecoach_trainer.settings import clear_settings_cache, storage_path


def test_metrics_count_selected_perspectives_and_duplicates(tmp_path, monkeypatch):
    import json

    from pokecoach_trainer.archive_catalog import catalog_connection

    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path, clock=SystemClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    cid = view.job.job_id
    with catalog_connection(tmp_path) as conn:
        conn.execute(
            """
            INSERT INTO campaign_replays(
                campaign_id, replay_id, decision, reason, filter_sha256, decision_json
            ) VALUES (?, 'a', 'selected', 'selected', 'fp', ?)
            """,
            (cid, json.dumps({"accepted_sides": ["p1"]})),
        )
        conn.execute(
            """
            INSERT INTO campaign_replays(
                campaign_id, replay_id, decision, reason, filter_sha256
            ) VALUES (?, 'b', 'rejected', 'duplicate_log', 'fp')
            """,
            (cid,),
        )
        conn.commit()
    progress = collection_progress(store, cid)
    assert progress["counters"]["selected_replays"] == 1
    assert progress["counters"]["accepted_perspectives"] == 1
    assert progress["counters"]["duplicate_replays"] == 1
    again = collection_progress(store, cid)
    assert again["counters"]["selected_replays"] == 1


def test_attempt_and_campaign_counters_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path, clock=SystemClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    progress = collection_progress(store, view.job.job_id)
    assert "selected_replays" in progress["counters"]
    assert progress["phase"] == "running"


def test_metrics_count_unique_transitions(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path, clock=SystemClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=5))
    store.command(view.job.job_id, "pause")
    progress = collection_progress(store, view.job.job_id)
    assert progress["phase"] == "paused"


def test_idle_discovery_and_provider_pause_are_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path, clock=SystemClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    progress = collection_progress(store, view.job.job_id)
    assert "discovery_windows_open" in progress
    assert progress.get("eta_s") is None


def test_eta_only_describes_known_queue(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path, clock=SystemClock())
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    progress = collection_progress(store, view.job.job_id)
    assert progress["eta_s"] is None


def test_log_tail_does_not_read_whole_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    log_path = storage_path("jobs", "job-log", "worker.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(f"line-{i}" for i in range(5000)), encoding="utf-8")
    lines = read_job_logs("job-log", tail=10)
    assert lines == [f"line-{i}" for i in range(4990, 5000)]

    worker_log = storage_path("jobs", "job-log", "workers", "w1.log")
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    worker_log.write_text("\n".join(f"w-{i}" for i in range(3000)), encoding="utf-8")
    lines_w = read_collection_worker_logs("job-log", "w1", tail=5)
    assert lines_w == [f"w-{i}" for i in range(2995, 3000)]
