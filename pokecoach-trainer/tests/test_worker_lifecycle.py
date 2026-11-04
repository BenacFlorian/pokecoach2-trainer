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


def test_only_one_long_job_across_types(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()

    hold_script = tmp_path / "hold_lock.py"
    pkg_root = Path(__file__).resolve().parents[1]
    hold_script.write_text(
        f"""
import os
import sys
import time
from pathlib import Path
root = Path({str(pkg_root)!r})
sys.path.insert(0, str(root / "src"))
os.chdir(root / "vendor" / "vgc-bench")
os.environ["TRAINER_STORAGE_ROOT"] = {str(tmp_path)!r}
from pokecoach_trainer.settings import clear_settings_cache
from pokecoach_trainer.jobs import acquire_job_lock
clear_settings_cache()
acquire_job_lock("blocking")
time.sleep(3)
""",
        encoding="utf-8",
    )
    blocker = subprocess.Popen(
        [sys.executable, str(hold_script)],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    try:
        time.sleep(0.5)
        cfg = CollectionJobConfig(config=CollectionConfig(format_ids=["mc"]))
        with pytest.raises(Conflict):
            submit_job(JobKind.collection, cfg)
    finally:
        blocker.wait(timeout=10)


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
