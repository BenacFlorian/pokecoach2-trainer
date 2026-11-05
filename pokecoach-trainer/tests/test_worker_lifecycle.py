import subprocess
import sys
import time
from pathlib import Path

import pytest

from pokecoach_trainer.errors import Conflict
from pokecoach_trainer.job_service import submit_job
from pokecoach_trainer.jobs import acquire_job_lock, get_job, release_job_lock, save_job
from pokecoach_trainer.schemas import (
    CollectionConfig,
    CollectionJobConfig,
    JobKind,
    JobRecord,
    JobStatus,
)


def test_collection_and_one_compute_job_can_coexist(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.jobs import acquire_resource_lock, release_resource_lock
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    cfd = acquire_resource_lock("collection", "camp-1")
    comp_fd = acquire_resource_lock("compute", "prep-1")
    release_resource_lock(comp_fd)
    release_resource_lock(cfd)


def test_second_compute_job_conflicts(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    fd_a = acquire_job_lock("job-a")
    with pytest.raises(RuntimeError):
        acquire_job_lock("job-b")
    release_job_lock("job-a", fd_a)


def test_queued_record_before_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    job_id = "job-test-1"
    record = JobRecord(
        job_id=job_id,
        kind=JobKind.collection,
        status=JobStatus.queued,
        config=CollectionJobConfig(config=CollectionConfig(format_ids=["mc"])),
    )
    save_job(record)
    loaded = get_job(job_id)
    assert loaded.status == JobStatus.queued
    fd = acquire_job_lock(job_id)
    release_job_lock(job_id, fd)


def test_import_prepare_keeps_compute_lock_after_api_restart(tmp_path, monkeypatch):
    """Le verrou compute reste pris tant que le détenteur ne libère pas explicitement."""
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    fd = acquire_job_lock("job-prepare-lock")
    try:
        with pytest.raises(RuntimeError):
            acquire_job_lock("job-other")
    finally:
        release_job_lock("job-prepare-lock", fd)
    fd2 = acquire_job_lock("job-after-release")
    release_job_lock("job-after-release", fd2)
