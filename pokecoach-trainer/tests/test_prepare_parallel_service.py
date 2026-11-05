"""Tests d'intégration prepare / workers (mocks pool + itérateur)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from pokecoach_trainer.archive_service import archive_replay, create_snapshot
from pokecoach_trainer.dataset_service import create_dataset, prepare_dataset
from pokecoach_trainer.prepare_resources import PrepareResources, resolve_conversion_workers
from pokecoach_trainer.schemas import (
    ArchiveSource,
    DatasetRecipe,
    PrepareExecutionConfig,
    SnapshotRequest,
)
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout
from prepare_process_helpers import FakeConversionExecutor, fake_converted_iter
from replay_log_fixtures import build_log


@pytest.fixture
def svc_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()


def _snap(n: int = 3) -> str:
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    for i in range(n):
        rid = f"gen9championsvgc2026regmc-svc{i}"
        archive_replay(
            rid,
            {
                "id": rid,
                "uploadtime": i,
                "log": build_log(p1_name=f"P{i}", winner=f"P{i}"),
                "format": "gen9championsvgc2026regmc",
            },
            source,
        )
    return create_snapshot(
        SnapshotRequest(format_ids=["gen9championsvgc2026regmc"])
    ).snapshot_id


def test_snapshot_prepare_uses_resolved_worker_count(svc_env, monkeypatch):
    snap = _snap(3)
    create_dataset(snap, DatasetRecipe(target_replays=3), dataset_id="ds-w")
    seen_workers: list[int] = []

    @contextmanager
    def recording_pool(*, workers, **kwargs):
        seen_workers.append(workers)
        yield FakeConversionExecutor(max_workers=workers)

    monkeypatch.setattr(
        "pokecoach_trainer.prepare_resources.detect_prepare_resources",
        lambda: PrepareResources(8, 8, 8, None, 8, ()),
    )
    with (
        patch("pokecoach_trainer.conversion_runtime.conversion_pool", side_effect=recording_pool),
        patch(
            "pokecoach_trainer.conversion.iter_converted_perspectives",
            fake_converted_iter,
        ),
    ):
        manifest = prepare_dataset(
            "ds-w",
            job_id="j1",
            execution=PrepareExecutionConfig(conversion_workers=4),
        )
    assert seen_workers
    assert manifest.prepare_execution is not None
    assert manifest.prepare_execution["requested_workers"] == 4
    assert manifest.prepare_execution["effective_workers"] == seen_workers[0]
    assert seen_workers[0] <= 4


def test_runtime_does_not_change_recipe_or_splits(svc_env):
    snap = _snap(3)
    recipe = DatasetRecipe(target_replays=3, seed=77)
    first = create_dataset(snap, recipe, dataset_id="ds-a")

    @contextmanager
    def pool(**kwargs):
        yield FakeConversionExecutor(max_workers=kwargs.get("workers", 1))

    with (
        patch("pokecoach_trainer.conversion_runtime.conversion_pool", side_effect=pool),
        patch(
            "pokecoach_trainer.conversion.iter_converted_perspectives",
            fake_converted_iter,
        ),
    ):
        ready = prepare_dataset(
            "ds-a",
            job_id="j2",
            execution=PrepareExecutionConfig(conversion_workers=1),
        )
    assert ready.recipe_sha256 == first.recipe_sha256
    assert ready.seed == recipe.seed


def test_no_pool_when_no_tasks(svc_env):
    assert (
        resolve_conversion_workers(
            "auto",
            PrepareResources(4, 4, 4, None, 4, ()),
            task_count=0,
        )
        == 0
    )
