"""Preuve mémoire bornée du tampon de mélange (processus isolé)."""

from __future__ import annotations

import multiprocessing as mp
import pickle
import resource
from pathlib import Path

import numpy as np
from imitation.data.types import Trajectory

from pokecoach_trainer.training_data import ShuffledTransitionBatches, TrajectorySplitSource
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation
from vgc_bench.src.utils import chunk_obs_len


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _synthetic_traj(steps: int, tag: float) -> Trajectory:
    obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
    obs[:, 0] = tag
    acts = np.zeros((steps, 2), dtype=np.int64)
    acts[:, 1] = 1
    return trajectory_to_imitation(
        Trajectory(obs=obs, acts=acts, infos=None, terminal=True),
    )


def _memory_worker(payload: tuple[int, int, int, Path]) -> tuple[int, int]:
    n_files, steps_per_file, shuffle_mib, tmp = payload
    from pokecoach_trainer.trajectory_dataset import load_trajectory_pickle

    files: list[Path] = []
    for i in range(n_files):
        path = tmp / f"t{i}.pkl"
        with path.open("wb") as f:
            pickle.dump(_synthetic_traj(steps_per_file, float(i)), f)
        files.append(path)

    def _loader(path: Path):
        return load_trajectory_pickle(path)

    import pokecoach_trainer.training_data as td

    td.load_trajectory_pickle = _loader  # type: ignore[misc]

    source = TrajectorySplitSource(
        split="train",
        files=tuple(files),
        expected_transitions=n_files * steps_per_file,
    )
    base = _rss_kib()
    shuffler = ShuffledTransitionBatches(
        source,
        batch_size=32,
        shuffle_buffer_mib=shuffle_mib,
        seed=1,
        epoch=1,
    )
    consumed = sum(b["acts"].shape[0] for b in shuffler)
    peak = _rss_kib()
    return base, peak if consumed == n_files * steps_per_file else -1


def test_shuffle_memory_sublinear_growth(tmp_path):
    ctx = mp.get_context("spawn")
    args = (64, 8, 32, tmp_path)
    with ctx.Pool(1) as pool:
        base_small, peak_small = pool.apply(_memory_worker, ((16, *args[1:]),))
        base_large, peak_large = pool.apply(_memory_worker, (args,))
    assert peak_small > 0 and peak_large > 0
    delta_small = peak_small - base_small
    delta_large = peak_large - base_large
    # La hausse RSS ne doit pas croître ~linéairement avec le nombre de fichiers.
    assert delta_large < delta_small * 4 + 50_000
