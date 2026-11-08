"""Tests sources de fichiers et batches séquentiels."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from imitation.data.types import DictObs, Trajectory

from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.training_data import (
    build_split_source,
    iter_source_trajectories,
    iter_transition_batches,
)
from pokecoach_trainer.trajectory_dataset import _obs_arrays, load_trajectory_pickle, trajectory_to_imitation
from training_fixtures import (
    install_storage_root,
    write_ready_dataset,
    write_ready_dataset_with_index,
)
from vgc_bench.src.utils import chunk_obs_len

_TERMINAL_SENTINEL = -999.0


def _tagged_traj(step_ids: list[int]) -> Trajectory:
    n_steps = len(step_ids)
    obs = np.zeros((n_steps + 1, 12 * chunk_obs_len), dtype=np.float32)
    for step, tid in enumerate(step_ids):
        obs[step, 0] = float(tid)
    obs[n_steps, 0] = _TERMINAL_SENTINEL
    acts = np.zeros((n_steps, 2), dtype=np.int64)
    acts[:, 1] = 1
    return trajectory_to_imitation(
        Trajectory(obs=obs, acts=acts, infos=None, terminal=True),
    )


def _manifest_for_dataset(dataset_id: str):
    from pokecoach_trainer.dataset_service import get_dataset

    return get_dataset(dataset_id)


def test_build_split_source_does_not_load_pickles(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    write_ready_dataset("ds-src", model, n_train=3, n_val=2, n_test=1)
    manifest = _manifest_for_dataset("ds-src")
    dataset_dir = storage_path("datasets", "ds-src")

    def _forbid_load(path):
        raise AssertionError(f"load_trajectory_pickle appelé: {path}")

    monkeypatch.setattr(
        "pokecoach_trainer.training_data.load_trajectory_pickle",
        _forbid_load,
    )
    source = build_split_source(dataset_dir, manifest, "train")
    assert source.expected_transitions == 9
    assert len(source.files) == 3
    assert list(source.files) == sorted(source.files)


def test_build_split_source_validates_index(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    write_ready_dataset_with_index(
        "ds-idx",
        model,
        train_steps=[2, 3],
        val_steps=[2],
        test_steps=[1],
        write_index=True,
    )
    manifest = _manifest_for_dataset("ds-idx")
    dataset_dir = storage_path("datasets", "ds-idx")
    build_split_source(dataset_dir, manifest, "train")

    index_path = dataset_dir / "trajectory_index.jsonl"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    bad = json.loads(lines[0])
    bad["transitions"] = 999
    index_path.write_text(json.dumps(bad) + "\n" + "\n".join(lines[1:]), encoding="utf-8")
    with pytest.raises(ValueError, match="transitions"):
        build_split_source(dataset_dir, manifest, "train")


def test_iter_source_trajectories_lazy_load(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    write_ready_dataset("ds-lazy", model, n_train=3, n_val=0, n_test=0, steps=2)
    manifest = _manifest_for_dataset("ds-lazy")
    source = build_split_source(storage_path("datasets", "ds-lazy"), manifest, "train")
    calls: list[str] = []

    def _track_load(path):
        calls.append(Path(path).name)
        return load_trajectory_pickle(path)

    monkeypatch.setattr(
        "pokecoach_trainer.training_data.load_trajectory_pickle",
        _track_load,
    )
    it = iter_source_trajectories(source)
    first = next(it)
    assert len(calls) == 1
    assert first is not None
    second = next(it)
    assert len(calls) == 2
    third = next(it)
    assert len(calls) == 3
    with pytest.raises(StopIteration):
        next(it)


def test_iter_source_trajectories_wraps_corruption(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    write_ready_dataset("ds-bad", model, n_train=1, n_val=0, n_test=0, steps=1)
    bad = storage_path("datasets", "ds-bad", "train", "t0.pkl")
    bad.write_bytes(b"not-a-pickle")
    manifest = _manifest_for_dataset("ds-bad")
    source = build_split_source(storage_path("datasets", "ds-bad"), manifest, "train")
    with pytest.raises(RuntimeError, match="lecture trajectoire"):
        list(iter_source_trajectories(source))


def test_iter_transition_batches_shapes_and_no_terminal_sentinel():
    trajs = [
        _tagged_traj([10, 11]),
        _tagged_traj([20, 21, 22]),
        _tagged_traj([30]),
    ]
    batches = list(iter_transition_batches(trajs, batch_size=4))
    assert [b["acts"].shape[0] for b in batches] == [4, 2]
    seen_ids: list[int] = []
    for batch in batches:
        assert isinstance(batch["obs"], DictObs)
        assert len(batch["obs"]) == batch["acts"].shape[0]
        assert batch["acts"].dtype == np.int64
        assert batch["acts"].shape[1] == 2
        obs = _obs_arrays(batch["obs"])["observation"]
        seen_ids.extend(int(obs[i, 0]) for i in range(obs.shape[0]))
    assert sorted(seen_ids) == [10, 11, 20, 21, 22, 30]
    assert _TERMINAL_SENTINEL not in seen_ids
    assert len(seen_ids) == len(set(seen_ids))
