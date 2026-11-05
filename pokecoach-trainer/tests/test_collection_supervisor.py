"""Tests superviseur collecte."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from pokecoach_trainer.collection_supervisor import (
    _is_process_alive,
    ensure_supervisor,
    run_collection_supervisor,
)
from pokecoach_trainer.jobs import acquire_resource_lock, release_resource_lock


def test_collection_and_compute_locks_are_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    cfd = acquire_resource_lock("collection", "camp-1")
    comp_fd = acquire_resource_lock("compute", "prep-1")
    release_resource_lock(comp_fd)
    release_resource_lock(cfd)


def test_two_launchers_spawn_one_supervisor(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    calls: list[int] = []

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    from pokecoach_trainer.settings import storage_path

    state_path = storage_path("jobs", "job-sup", "supervisor.json")

    def is_alive(*_args):
        return state_path.is_file()

    with mock.patch(
        "pokecoach_trainer.collection_supervisor._is_process_alive", side_effect=is_alive
    ), mock.patch(
        "pokecoach_trainer.collection_supervisor.acquire_resource_lock", return_value=9
    ), mock.patch(
        "pokecoach_trainer.collection_supervisor.subprocess.Popen", side_effect=lambda *a, **k: calls.append(1) or FakeProc()
    ), mock.patch("pokecoach_trainer.collection_supervisor.os.close"):
        ensure_supervisor(tmp_path, "job-sup")
        ensure_supervisor(tmp_path, "job-sup")
    assert len(calls) == 1


def test_api_exit_does_not_release_child_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    released: list[int] = []

    with mock.patch(
        "pokecoach_trainer.collection_supervisor.release_resource_lock",
        side_effect=lambda fd: released.append(fd),
    ), mock.patch(
        "pokecoach_trainer.collection_supervisor._scale_children"
    ), mock.patch(
        "pokecoach_trainer.collection_supervisor.get_job"
    ) as get_job, mock.patch(
        "pokecoach_trainer.collection_supervisor.save_job"
    ), mock.patch(
        "pokecoach_trainer.collection_supervisor.CollectionStore"
    ) as store_cls:
        store = store_cls.return_value
        store.get_campaign.return_value.phase = "stopped"
        store.get_campaign.return_value.effective_config.shutdown_grace_s = 0
        get_job.return_value.status = "running"
        run_collection_supervisor(tmp_path, "job-x", lock_fd=7)
    assert released == [7]


def test_pid_reuse_is_not_live_worker():
    assert _is_process_alive(999999, "boot", 0) is False


def test_dead_supervisor_is_reconciled_without_duplicate_children(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout, storage_path

    clear_settings_cache()
    ensure_storage_layout()
    state_path = storage_path("jobs", "job-dead", "supervisor.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"pid": 1, "boot_id": "x", "start_ticks": 0}), encoding="utf-8")
    spawned: list[int] = []

    class FakeProc:
        pid = 5151

        def poll(self):
            return None

    with mock.patch(
        "pokecoach_trainer.collection_supervisor._is_process_alive", return_value=False
    ), mock.patch(
        "pokecoach_trainer.collection_supervisor.acquire_resource_lock", return_value=3
    ), mock.patch(
        "pokecoach_trainer.collection_supervisor.subprocess.Popen",
        side_effect=lambda *a, **k: spawned.append(1) or FakeProc(),
    ), mock.patch("pokecoach_trainer.collection_supervisor.os.close"):
        ensure_supervisor(tmp_path, "job-dead")
    assert len(spawned) == 1


def test_scale_changes_pool_without_new_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.collection_supervisor import _scale_children
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    children: dict = {}
    with mock.patch("pokecoach_trainer.collection_supervisor.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        _scale_children(tmp_path, "job-scale", children, discovery=2, download=1)
        assert len(children) == 3
        _scale_children(tmp_path, "job-scale", children, discovery=1, download=1)
        assert len(children) == 2
