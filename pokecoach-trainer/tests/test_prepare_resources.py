"""Tests prepare_resources (sans modifier le cgroup hôte)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokecoach_trainer.prepare_resources import (
    PrepareResources,
    detect_prepare_resources,
    resolve_conversion_workers,
)


def _fake_read(files: dict[str, str]):
    def _read(path: Path) -> str | None:
        key = str(path)
        for pattern, content in files.items():
            if key.endswith(pattern) or key == pattern:
                return content
        return None

    return _read


def test_host64_affinity8_cap8():
    resources = detect_prepare_resources(
        read_text=_fake_read({}),
        process_cpu_count=lambda: 64,
        sched_getaffinity=lambda _: set(range(8)),
        cpu_count=lambda: 64,
    )
    assert resources.available_workers == 8


def test_affinity8_quota4_cap4():
    files = {
        "/proc/self/cgroup": "0::/slice/job\n",
        "/proc/self/mountinfo": (
            "999 998 0:37 / /sys/fs/cgroup rw shared:1 - cgroup2 none rw\n"
        ),
        "cpu.max": "400000 100000\n",
    }
    resources = detect_prepare_resources(
        read_text=_fake_read(files),
        process_cpu_count=lambda: 8,
        sched_getaffinity=lambda _: set(range(8)),
        cpu_count=lambda: 8,
    )
    assert resources.quota_cpus == 4.0
    assert resources.available_workers == 4


def test_quota_two_and_half_yields_three_workers():
    files = {
        "/proc/self/cgroup": "0::/child\n",
        "/proc/self/mountinfo": "1 0 0:37 / /cgroup rw - cgroup2 none rw\n",
        "cpu.max": "250000 100000\n",
    }
    resources = detect_prepare_resources(
        read_text=_fake_read(files),
        process_cpu_count=lambda: 8,
        sched_getaffinity=lambda _: set(range(8)),
        cpu_count=lambda: 8,
    )
    assert resources.available_workers == 3


def test_nested_cgroup_child_unlimited_parent_limits():
    def read(path: Path) -> str | None:
        s = str(path)
        if s.endswith("/proc/self/cgroup"):
            return "0::/parent/child\n"
        if s.endswith("/proc/self/mountinfo"):
            return "1 0 0:37 / /cg rw - cgroup2 none rw\n"
        if s.endswith("/parent/child/cpu.max"):
            return "max 100000\n"
        if s.endswith("/parent/cpu.max"):
            return "200000 100000\n"
        if s.endswith("/cpu.max"):
            return "max 100000\n"
        return None

    resources = detect_prepare_resources(
        read_text=read,
        process_cpu_count=lambda: 8,
        sched_getaffinity=lambda _: set(range(8)),
        cpu_count=lambda: 8,
    )
    assert resources.quota_cpus == 2.0
    assert resources.available_workers == 2


def test_all_unavailable_fallback_one():
    resources = detect_prepare_resources(
        read_text=_fake_read({}),
        process_cpu_count=lambda: None,
        sched_getaffinity=lambda _: (_ for _ in ()).throw(OSError()),
        cpu_count=lambda: None,
    )
    assert resources.available_workers == 1


def test_resolve_task_count_zero():
    res = PrepareResources(8, 8, 8, None, 8, ())
    assert resolve_conversion_workers("auto", res, task_count=0) == 0


def test_resolve_machine8_tasks3():
    res = PrepareResources(8, 8, 8, None, 8, ())
    assert resolve_conversion_workers("auto", res, task_count=3) == 3


def test_resolve_explicit4_on_cap2():
    res = PrepareResources(2, 2, 2, None, 2, ())
    assert resolve_conversion_workers(4, res, task_count=10) == 2


def test_resolve_explicit1_stays_one():
    res = PrepareResources(8, 8, 8, None, 8, ())
    assert resolve_conversion_workers(1, res, task_count=5) == 1


def test_cgroup_v1_quota():
    files = {
        "/proc/self/cgroup": "11:cpu,cpuacct:/user.slice/job\n",
        "/proc/self/mountinfo": (
            "34 23 0:29 / /sys/fs/cgroup/cpu rw shared:1 - cgroup cpu rw\n"
        ),
        "cpu.cfs_quota_us": "200000\n",
        "cpu.cfs_period_us": "100000\n",
    }
    resources = detect_prepare_resources(
        read_text=_fake_read(files),
        process_cpu_count=lambda: 8,
        sched_getaffinity=lambda _: set(range(8)),
        cpu_count=lambda: 8,
    )
    assert resources.quota_cpus == 2.0
    assert resources.available_workers == 2


def test_malformed_cpu_max_ignored():
    files = {
        "/proc/self/cgroup": "0::/x\n",
        "/proc/self/mountinfo": "1 0 0:37 / /cg rw - cgroup2 none rw\n",
        "cpu.max": "not-a-quota\n",
    }
    resources = detect_prepare_resources(
        read_text=_fake_read(files),
        process_cpu_count=lambda: 4,
        sched_getaffinity=lambda _: set(range(4)),
        cpu_count=lambda: 4,
    )
    assert resources.quota_cpus is None
    assert resources.available_workers == 4
