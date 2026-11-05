from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest
from imitation.data.types import Trajectory
from vgc_bench.src.utils import chunk_obs_len

from pokecoach_trainer.archive_service import archive_replay, create_snapshot
from pokecoach_trainer.dataset_service import create_dataset, prepare_dataset
from pokecoach_trainer.schemas import (
    ArchiveSource,
    ConversionResult,
    ConversionStatus,
    DatasetRecipe,
    DatasetState,
    PrepareExecutionConfig,
    SnapshotRequest,
)
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout, storage_path
from replay_log_fixtures import build_log
from prepare_process_helpers import fake_converted_iter, fake_conversion_pool


@pytest.fixture
def ds_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield


def _fake_traj(steps: int = 2) -> Trajectory:
    obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
    acts = np.zeros((steps, 2), dtype=np.int64)
    return Trajectory(obs=obs, acts=acts, infos=None, terminal=True)


def _seed_archive(replay_ids: list[str]) -> str:
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    for idx, replay_id in enumerate(replay_ids):
        log = build_log(
            p1_name=f"Player{idx}",
            p2_name=f"Opp{idx}",
            p1_rating=1300,
            p2_rating=1300,
            winner=f"Player{idx}",
        )
        archive_replay(
            replay_id,
            {
                "id": replay_id,
                "uploadtime": idx,
                "log": log,
                "format": "gen9championsvgc2026regmc",
            },
            source,
        )
    snap = create_snapshot(SnapshotRequest(format_ids=["gen9championsvgc2026regmc"]))
    return snap.snapshot_id


def test_ready_dataset_immutable(ds_env):
    snap_id = _seed_archive(
        [f"gen9championsvgc2026regmc-r{i}" for i in range(3)],
    )
    recipe = DatasetRecipe(target_replays=3)
    manifest = create_dataset(snap_id, recipe, dataset_id="ds-immutable")
    with (
        patch(
            "pokecoach_trainer.conversion_runtime.conversion_pool",
            fake_conversion_pool,
        ),
        patch(
            "pokecoach_trainer.conversion.iter_converted_perspectives",
            fake_converted_iter,
        ),
    ):
        ready = prepare_dataset(
            "ds-immutable",
            job_id="job-test",
            execution=PrepareExecutionConfig(conversion_workers=1),
        )
    assert ready.state == DatasetState.ready
    with pytest.raises(FileExistsError):
        create_dataset(snap_id, recipe, dataset_id="ds-immutable")


def test_insufficient_data_when_target_not_met(ds_env):
    snap_id = _seed_archive(["gen9championsvgc2026regmc-r2"])
    recipe = DatasetRecipe(target_replays=5)
    create_dataset(snap_id, recipe, dataset_id="ds-insuff")
    result = prepare_dataset("ds-insuff", job_id="job-insuff")
    assert result.state == DatasetState.insufficient_data
    assert not (storage_path("datasets", "ds-insuff", "train").exists())


def test_splits_disjoint_on_disk(ds_env):
    replay_ids = [f"gen9championsvgc2026regmc-s{i}" for i in range(3)]
    snap_id = _seed_archive(replay_ids)
    recipe = DatasetRecipe(target_replays=3)
    create_dataset(snap_id, recipe, dataset_id="ds-splits")

    with (
        patch(
            "pokecoach_trainer.conversion_runtime.conversion_pool",
            fake_conversion_pool,
        ),
        patch(
            "pokecoach_trainer.conversion.iter_converted_perspectives",
            fake_converted_iter,
        ),
    ):
        manifest = prepare_dataset(
            "ds-splits",
            job_id="job-splits",
            execution=PrepareExecutionConfig(conversion_workers=2),
        )
    assert manifest.state == DatasetState.ready
    train = set(manifest.replay_ids_train)
    val = set(manifest.replay_ids_val)
    test = set(manifest.replay_ids_test)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
