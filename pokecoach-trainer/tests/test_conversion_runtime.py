"""Tests du pool de conversion (spawn, initialisation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokecoach_trainer.conversion_runtime import conversion_pool


def test_pool_uses_spawn_and_requested_workers(monkeypatch, tmp_path):
    captured: dict = {}

    class RecordingExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def shutdown(self, **_kwargs):
            return None

    monkeypatch.setattr(
        "pokecoach_trainer.conversion_runtime.ProcessPoolExecutor",
        RecordingExecutor,
    )
    root = tmp_path / "storage"
    vendor = tmp_path / "vendor"
    root.mkdir()
    vendor.mkdir()
    with conversion_pool(
        workers=3,
        storage_root=root,
        vendor_root=vendor,
        timeout_s=30.0,
    ):
        pass
    assert captured["max_workers"] == 3
    assert captured["mp_context"].get_start_method() == "spawn"
    assert captured["initializer"].__name__ == "initialize_conversion_worker"


def test_pool_zero_workers_yields_none(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    with conversion_pool(
        workers=0,
        storage_root=root,
        vendor_root=root,
        timeout_s=1.0,
    ) as executor:
        assert executor is None
