"""Tests mélange borné ShuffledTransitionBatches."""

from __future__ import annotations

import pickle
from collections import Counter

import numpy as np
import pytest

from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.training_data import (
    ShuffledTransitionBatches,
    build_split_source,
    iter_transition_batches,
)
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation
from training_fixtures import install_storage_root, write_ready_dataset_with_index
from imitation.data.types import Trajectory
from vgc_bench.src.utils import chunk_obs_len

_TERMINAL = -888.0


def _tagged_traj(ids: list[int]) -> Trajectory:
    n = len(ids)
    obs = np.zeros((n + 1, 12 * chunk_obs_len), dtype=np.float32)
    for i, tid in enumerate(ids):
        obs[i, 0] = float(tid)
        obs[i, 1] = float(tid)
    obs[n, 0] = _TERMINAL
    acts = np.zeros((n, 2), dtype=np.int64)
    acts[:, 1] = 1
    return trajectory_to_imitation(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))


def _write_tagged_dataset(dataset_id: str, step_lists: list[list[int]]) -> None:
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    flat_steps = [len(s) for s in step_lists]
    write_ready_dataset_with_index(dataset_id, model, train_steps=flat_steps, val_steps=[], test_steps=[])
    d = storage_path("datasets", dataset_id, "train")
    for i, ids in enumerate(step_lists):
        traj = _tagged_traj(ids)
        with (d / f"t{i}.pkl").open("wb") as f:
            pickle.dump(traj, f)


def _train_source(dataset_id: str):
    from pokecoach_trainer.dataset_service import get_dataset

    manifest = get_dataset(dataset_id)
    return build_split_source(storage_path("datasets", dataset_id), manifest, "train")


def _collect_transition_ids(shuffler: ShuffledTransitionBatches) -> list[int]:
    ids: list[int] = []
    for batch in shuffler:
        from pokecoach_trainer.trajectory_dataset import _obs_arrays

        obs = _obs_arrays(batch["obs"])["observation"]
        ids.extend(int(obs[j, 0]) for j in range(obs.shape[0]))
    return ids


@pytest.mark.parametrize("n", [1, 31, 32, 33])
def test_shuffled_exact_coverage(n: int, tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    dataset_id = f"ds-shuf-{n}"
    _write_tagged_dataset(dataset_id, [list(range(n))])
    source = _train_source(dataset_id)
    shuffler = ShuffledTransitionBatches(
        source,
        batch_size=32,
        shuffle_buffer_mib=256,
        seed=7,
        epoch=1,
    )
    out_ids = _collect_transition_ids(shuffler)
    assert Counter(out_ids) == Counter(range(n))
    assert len(out_ids) == n


def test_shuffled_deterministic_same_seed_epoch(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    _write_tagged_dataset("ds-det", [list(range(20)), list(range(100, 115))])
    source = _train_source("ds-det")
    a = _collect_transition_ids(
        ShuffledTransitionBatches(source, batch_size=8, shuffle_buffer_mib=256, seed=3, epoch=1),
    )
    b = _collect_transition_ids(
        ShuffledTransitionBatches(source, batch_size=8, shuffle_buffer_mib=256, seed=3, epoch=1),
    )
    c = _collect_transition_ids(
        ShuffledTransitionBatches(source, batch_size=8, shuffle_buffer_mib=256, seed=3, epoch=2),
    )
    assert a == b
    assert a != c


def test_shuffled_batches_not_shared_with_sources(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    _write_tagged_dataset("ds-mem", [list(range(12))])
    source = _train_source("ds-mem")
    shuffler = ShuffledTransitionBatches(
        source,
        batch_size=4,
        shuffle_buffer_mib=64,
        seed=1,
        epoch=1,
    )
    it = iter(shuffler)
    first = next(it)
    from pokecoach_trainer.trajectory_dataset import _obs_arrays

    src_obs = _obs_arrays(first["obs"])["observation"]
    second = next(it)
    _obs_arrays(second["obs"])["observation"][0, 0] = 12345.0
    assert src_obs[0, 0] != 12345.0
    assert not np.shares_memory(src_obs, _obs_arrays(second["obs"])["observation"])


def test_shuffled_buffer_bounded_numpy_bytes(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    _write_tagged_dataset("ds-budget", [list(range(50))])
    source = _train_source("ds-budget")
    mib = 4
    budget = mib * 1024**2
    shuffler = ShuffledTransitionBatches(
        source,
        batch_size=16,
        shuffle_buffer_mib=mib,
        seed=0,
        epoch=1,
    )
    gen = shuffler._run()
    next(gen)
    nbytes = 0
    if shuffler._observation is not None:
        nbytes += shuffler._observation.nbytes
    if shuffler._action_mask is not None:
        nbytes += shuffler._action_mask.nbytes
    if shuffler._acts is not None:
        nbytes += shuffler._acts.nbytes
    assert nbytes <= budget


def test_sequential_batches_independent():
    trajs = [_tagged_traj([1, 2, 3, 4, 5])]
    it = iter_transition_batches(trajs, batch_size=3)
    kept = next(it)
    nxt = next(it)
    nxt["acts"][0, 0] = 99
    assert kept["acts"][0, 0] != 99
