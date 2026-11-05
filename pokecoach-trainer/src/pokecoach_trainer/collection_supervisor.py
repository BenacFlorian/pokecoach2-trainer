"""Superviseur de campagne collecte (pools découverte/téléchargement)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.collection_logging import format_progress_snapshot, log_campaign
from pokecoach_trainer.collection_metrics import collection_progress
from pokecoach_trainer.jobs import acquire_resource_lock, get_job, release_resource_lock, save_job
from pokecoach_trainer.schemas import JobStatus
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import write_json_atomic


def _supervisor_state_path(campaign_id: str) -> Path:
    return storage_path("jobs", campaign_id, "supervisor.json")


def _is_process_alive(pid: int, boot_id: str, start_ticks: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.is_file():
        return pid > 0
    parts = stat_path.read_text(encoding="utf-8").split()
    if len(parts) < 22:
        return True
    try:
        proc_start = int(parts[21])
    except ValueError:
        return True
    return proc_start == start_ticks


def ensure_supervisor(root: Path, campaign_id: str) -> None:
    """Démarre ou rejoint le superviseur détaché pour une campagne."""

    state_path = _supervisor_state_path(campaign_id)
    if state_path.is_file():
        meta = json.loads(state_path.read_text(encoding="utf-8"))
        if _is_process_alive(
            int(meta.get("pid", 0)),
            str(meta.get("boot_id", "")),
            int(meta.get("start_ticks", 0)),
        ):
            return
    fd = acquire_resource_lock("collection", campaign_id)
    package_root = Path(__file__).resolve().parents[2]
    log_path = storage_path("jobs", campaign_id, "worker.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "pokecoach_trainer.worker",
                "--role",
                "collection-supervisor",
                "--campaign-id",
                campaign_id,
                "--lock-fd",
                str(fd),
            ],
            cwd=str(package_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            pass_fds=(fd,),
            start_new_session=True,
            env=child_env,
        )
    os.close(fd)
    write_json_atomic(
        state_path,
        {"pid": proc.pid, "boot_id": _read_boot_id(), "start_ticks": _proc_start_ticks(proc.pid)},
    )


def _read_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _proc_start_ticks(pid: int) -> int:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.is_file():
        return 0
    parts = stat_path.read_text(encoding="utf-8").split()
    if len(parts) < 22:
        return 0
    try:
        return int(parts[21])
    except ValueError:
        return 0


def run_collection_supervisor(root: Path, campaign_id: str, *, lock_fd: int) -> int:
    """Boucle superviseur : lance les workers et projette state.json."""

    store = CollectionStore(root)
    children: dict[str, subprocess.Popen] = {}
    last_progress_log = 0.0
    log_campaign(campaign_id, "superviseur démarré")
    try:
        while True:
            view = store.get_campaign(campaign_id)
            record = get_job(campaign_id)
            record.status = JobStatus.running if view.phase in {
                "running",
                "queued",
                "pausing",
            } else JobStatus.completed
            progress = collection_progress(store, campaign_id)
            record.progress = progress
            save_job(record)
            now = time.monotonic()
            if now - last_progress_log >= 10.0:
                log_campaign(campaign_id, format_progress_snapshot(progress))
                last_progress_log = now
            if view.phase in {"completed", "stopped", "failed", "budget_reached", "interrupted"}:
                log_campaign(campaign_id, f"arrêt phase={view.phase} reason={view.stop_reason}")
                break
            desired_d = view.effective_config.discovery_workers
            desired_dl = view.effective_config.download_workers
            _scale_children(
                root,
                campaign_id,
                children,
                discovery=desired_d,
                download=desired_dl,
            )
            time.sleep(1.0)
    finally:
        grace = store.get_campaign(campaign_id).effective_config.shutdown_grace_s
        deadline = time.monotonic() + grace
        for proc in list(children.values()):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        while time.monotonic() < deadline:
            if all(p.poll() is not None for p in children.values()):
                break
            time.sleep(0.2)
        for proc in children.values():
            if proc.poll() is None:
                proc.kill()
        release_resource_lock(lock_fd)
    return 0


def _scale_children(
    root: Path,
    campaign_id: str,
    children: dict[str, subprocess.Popen],
    *,
    discovery: int,
    download: int,
) -> None:
    package_root = Path(__file__).resolve().parents[2]
    targets: list[tuple[str, str]] = []
    for i in range(discovery):
        targets.append((f"disc-{i}", "discovery"))
    for i in range(download):
        targets.append((f"dl-{i}", "download"))
    target_ids = {wid for wid, _ in targets}
    for wid, proc in list(children.items()):
        if wid not in target_ids and proc.poll() is None:
            proc.terminate()
            del children[wid]
    for wid, role in targets:
        if wid in children and children[wid].poll() is None:
            continue
        log = storage_path("jobs", campaign_id, "workers", f"{wid}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with log.open("a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "pokecoach_trainer.worker",
                    "--role",
                    "collection-worker",
                    "--campaign-id",
                    campaign_id,
                    "--worker-id",
                    wid,
                    "--worker-kind",
                    role,
                ],
                cwd=str(package_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
        children[wid] = proc
