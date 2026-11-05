"""Détection CPU/cgroup pour la préparation parallèle."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PrepareResources:
    logical_cpus: int | None
    affinity_cpus: int | None
    process_cpus: int | None
    quota_cpus: float | None
    available_workers: int
    detection_notes: tuple[str, ...]


def _positive_or_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_cgroup_v2_cpu_max(text: str) -> float | None:
    parts = text.strip().split()
    if len(parts) != 2:
        return None
    quota_raw, period_raw = parts
    if quota_raw == "max":
        return None
    try:
        quota = int(quota_raw)
        period = int(period_raw)
    except ValueError:
        return None
    if period <= 0:
        return None
    return quota / period


def _parse_cgroup_v1_cpu(text_quota: str, text_period: str) -> float | None:
    try:
        quota = int(text_quota.strip())
        period = int(text_period.strip())
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _cgroup_v2_mount_root(mountinfo: str) -> Path | None:
    for line in mountinfo.splitlines():
        if "- cgroup2 " not in line:
            continue
        dash = line.find(" - ")
        if dash < 0:
            continue
        left = line[:dash].split()
        if len(left) >= 5:
            return Path(left[4])
    return None


def _cgroup_v1_cpu_mount(mountinfo: str) -> Path | None:
    for line in mountinfo.splitlines():
        if "- cgroup " not in line:
            continue
        dash = line.find(" - ")
        if dash < 0:
            continue
        left = line[:dash].split()
        if len(left) < 5:
            continue
        mount_point = left[4]
        if "cpu" not in mount_point:
            continue
        return Path(mount_point)
    return None


def _relative_cgroup_path(cgroup_self: str) -> tuple[str | None, str | None]:
    """Retourne (chemin cgroup v2, chemin cgroup v1 cpu) si présents."""

    v2_path: str | None = None
    v1_cpu: str | None = None
    for line in cgroup_self.strip().splitlines():
        if not line.strip():
            continue
        _id, controllers, path = line.split(":", 2)
        if controllers == "" and path:
            v2_path = path.lstrip("/")
        if "cpu" in controllers.split(",") and path:
            v1_cpu = path.lstrip("/")
    return v2_path, v1_cpu


def _walk_ancestors(rel_path: str) -> list[str]:
    rel_path = rel_path.strip("/")
    if not rel_path:
        return [""]
    parts = rel_path.split("/")
    paths = [""]
    for i in range(len(parts)):
        paths.append("/".join(parts[: i + 1]))
    return paths


def _finite_quota_from_cgroup_v2(
    mount_root: Path,
    rel_path: str,
    *,
    read_text=_read_text,
) -> float | None:
    quotas: list[float] = []
    for segment in _walk_ancestors(rel_path):
        cpu_max = mount_root / segment / "cpu.max"
        text = read_text(cpu_max)
        if text is None:
            continue
        parsed = _parse_cgroup_v2_cpu_max(text)
        if parsed is not None:
            quotas.append(parsed)
    if not quotas:
        return None
    return min(quotas)


def _finite_quota_from_cgroup_v1(
    mount_root: Path,
    rel_path: str,
    *,
    read_text=_read_text,
) -> float | None:
    quotas: list[float] = []
    for segment in _walk_ancestors(rel_path):
        base = mount_root / segment
        q_text = read_text(base / "cpu.cfs_quota_us")
        p_text = read_text(base / "cpu.cfs_period_us")
        if q_text is None or p_text is None:
            continue
        parsed = _parse_cgroup_v1_cpu(q_text, p_text)
        if parsed is not None:
            quotas.append(parsed)
    if not quotas:
        return None
    return min(quotas)


def detect_prepare_resources(
    *,
    read_text=_read_text,
    process_cpu_count=os.process_cpu_count if hasattr(os, "process_cpu_count") else None,
    sched_getaffinity=os.sched_getaffinity if hasattr(os, "sched_getaffinity") else None,
    cpu_count=os.cpu_count,
) -> PrepareResources:
    notes: list[str] = []
    logical = _positive_or_none(cpu_count())
    process_cpus = None
    if process_cpu_count is not None:
        try:
            process_cpus = _positive_or_none(process_cpu_count())
        except OSError:
            notes.append("process_cpu_count a échoué")
    affinity = None
    if sched_getaffinity is not None:
        try:
            affinity = _positive_or_none(len(sched_getaffinity(0)))
        except OSError:
            notes.append("sched_getaffinity a échoué")

    host_limits = [
        x
        for x in (process_cpus, affinity, logical)
        if x is not None
    ]
    host_cap = min(host_limits) if host_limits else None

    cgroup_self = read_text(Path("/proc/self/cgroup"))
    mountinfo = read_text(Path("/proc/self/mountinfo"))
    quota_cpus: float | None = None
    if cgroup_self and mountinfo:
        v2_rel, v1_rel = _relative_cgroup_path(cgroup_self)
        v2_root = _cgroup_v2_mount_root(mountinfo)
        if v2_root is not None and v2_rel is not None:
            quota_cpus = _finite_quota_from_cgroup_v2(
                v2_root, v2_rel, read_text=read_text
            )
        if quota_cpus is None and v1_rel is not None:
            v1_root = _cgroup_v1_cpu_mount(mountinfo)
            if v1_root is not None:
                quota_cpus = _finite_quota_from_cgroup_v1(
                    v1_root, v1_rel, read_text=read_text
                )

    if host_cap is None and quota_cpus is None:
        available = 1
        notes.append("repli ultime: 1 worker")
    elif quota_cpus is not None:
        quota_workers = max(1, math.ceil(quota_cpus))
        if host_cap is not None:
            available = min(host_cap, quota_workers)
            if quota_workers > host_cap:
                notes.append("quota cgroup plafonné par affinité/process CPUs")
        else:
            available = quota_workers
    else:
        available = host_cap or 1

    return PrepareResources(
        logical_cpus=logical,
        affinity_cpus=affinity,
        process_cpus=process_cpus,
        quota_cpus=quota_cpus,
        available_workers=available,
        detection_notes=tuple(notes),
    )


def resolve_conversion_workers(
    requested: Literal["auto"] | int,
    resources: PrepareResources,
    *,
    task_count: int,
) -> int:
    if task_count <= 0:
        return 0
    cap = max(1, resources.available_workers)
    if requested == "auto":
        chosen = cap
    else:
        chosen = min(int(requested), cap)
    return min(chosen, task_count)
