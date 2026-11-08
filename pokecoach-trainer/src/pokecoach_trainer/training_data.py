"""Sources de trajectoires, batches de transitions et mélange borné."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from imitation.data.types import DictObs, Trajectory

from pokecoach_trainer.schemas import DatasetManifest
from pokecoach_trainer.trajectory_dataset import _obs_arrays, load_trajectory_pickle

SplitName = Literal["train", "validation", "test"]

SAMPLER_VERSION = "file_buffer_shuffle/1"

_SPLIT_DIR = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}

_SPLIT_MANIFEST_FIELDS: dict[
    SplitName, tuple[str, str]
] = {
    "train": ("trajectories_train", "transitions_train"),
    "validation": ("trajectories_val", "transitions_val"),
    "test": ("trajectories_test", "transitions_test"),
}


class TransitionBatch(TypedDict):
    obs: DictObs
    acts: np.ndarray


@dataclass(frozen=True, slots=True)
class TrajectorySplitSource:
    split: SplitName
    files: tuple[Path, ...]
    expected_transitions: int


def build_split_source(
    dataset_dir: Path,
    manifest: DatasetManifest,
    split: SplitName,
) -> TrajectorySplitSource:
    """Construit une source de chemins sans désérialiser les pickles."""

    split_folder = _SPLIT_DIR[split]
    directory = dataset_dir / split_folder
    files = tuple(sorted(directory.glob("*.pkl")))
    traj_field, trans_field = _SPLIT_MANIFEST_FIELDS[split]
    expected_traj = int(getattr(manifest, traj_field))
    expected_trans = int(getattr(manifest, trans_field))
    if len(files) != expected_traj:
        raise ValueError(
            f"fichiers {split}: {len(files)} != manifest {expected_traj}",
        )
    _validate_trajectory_index(dataset_dir, split, files, expected_trans)
    return TrajectorySplitSource(
        split=split,
        files=files,
        expected_transitions=expected_trans,
    )


def _validate_trajectory_index(
    dataset_dir: Path,
    split: SplitName,
    files: tuple[Path, ...],
    expected_transitions: int,
) -> None:
    index_path = dataset_dir / "trajectory_index.jsonl"
    if not index_path.is_file():
        return
    expected_names = {p.name for p in files}
    seen: set[str] = set()
    indexed_names: set[str] = set()
    transition_sum = 0
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row_split = row.get("split")
            if row_split != split:
                continue
            fname = row.get("file")
            if not isinstance(fname, str):
                raise ValueError("index: champ file invalide")
            if fname in seen:
                raise ValueError(f"index: doublon {fname}")
            seen.add(fname)
            file_path = dataset_dir / _SPLIT_DIR[split] / fname
            if not file_path.is_file():
                raise ValueError(f"index: fichier absent {fname}")
            indexed_names.add(fname)
            transition_sum += int(row.get("transitions", 0))
    if indexed_names != expected_names:
        missing = expected_names - indexed_names
        extra = indexed_names - expected_names
        if missing or extra:
            raise ValueError(
                f"index {split}: écart fichiers missing={missing} extra={extra}",
            )
    if transition_sum != expected_transitions:
        raise ValueError(
            f"index {split}: transitions {transition_sum} != manifest {expected_transitions}",
        )


def iter_source_trajectories(
    source: TrajectorySplitSource,
    *,
    file_order: Sequence[int] | None = None,
) -> Iterator[Trajectory]:
    """Charge les trajectoires une à une, dans l'ordre demandé."""

    order = range(len(source.files)) if file_order is None else file_order
    for idx in order:
        if idx < 0 or idx >= len(source.files):
            raise IndexError(f"indice fichier hors bornes: {idx}")
        path = source.files[idx]
        try:
            yield load_trajectory_pickle(path)
        except Exception as exc:
            raise RuntimeError(f"lecture trajectoire {path}") from exc


def _transition_rows(traj: Trajectory) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_steps = len(traj.acts)
    if n_steps == 0:
        raise ValueError("trajectoire vide")
    obs_map = _obs_arrays(traj.obs)
    observation = np.asarray(obs_map["observation"])
    action_mask = np.asarray(obs_map["action_mask"])
    if observation.shape[0] != n_steps + 1 or action_mask.shape[0] != n_steps + 1:
        raise ValueError("formes observation/masque incohérentes")
    acts = np.asarray(traj.acts, dtype=np.int64)
    if acts.shape != (n_steps, 2):
        raise ValueError("forme actions invalide")
    for step in range(n_steps):
        yield (
            np.asarray(observation[step], dtype=observation.dtype).copy(),
            np.asarray(action_mask[step], dtype=action_mask.dtype).copy(),
            acts[step].copy(),
        )


def _owned_batch(
    obs_rows: list[np.ndarray],
    mask_rows: list[np.ndarray],
    act_rows: list[np.ndarray],
) -> TransitionBatch:
    observation = np.stack(obs_rows, axis=0)
    action_mask = np.stack(mask_rows, axis=0)
    acts = np.stack(act_rows, axis=0).astype(np.int64, copy=False)
    return TransitionBatch(
        obs=DictObs(
            {
                "observation": np.ascontiguousarray(observation),
                "action_mask": np.ascontiguousarray(action_mask),
            },
        ),
        acts=np.ascontiguousarray(acts),
    )


def iter_transition_batches(
    trajectories: Iterable[Trajectory],
    *,
    batch_size: int,
) -> Iterator[TransitionBatch]:
    """Émet des batches de transitions (lecture séquentielle, sans mélange)."""

    if batch_size < 1:
        raise ValueError("batch_size invalide")
    obs_buf: list[np.ndarray] = []
    mask_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []

    def flush() -> TransitionBatch | None:
        if not obs_buf:
            return None
        batch = _owned_batch(obs_buf, mask_buf, act_buf)
        obs_buf.clear()
        mask_buf.clear()
        act_buf.clear()
        return batch

    for traj in trajectories:
        for obs_row, mask_row, act_row in _transition_rows(traj):
            obs_buf.append(obs_row)
            mask_buf.append(mask_row)
            act_buf.append(act_row)
            if len(obs_buf) >= batch_size:
                batch = flush()
                assert batch is not None
                yield batch
        del traj
    tail = flush()
    if tail is not None:
        yield tail


def _row_bytes_from_traj(traj: Trajectory) -> int:
    for obs_row, mask_row, _ in _transition_rows(traj):
        return int(obs_row.nbytes + mask_row.nbytes + 2 * np.dtype(np.int64).itemsize)
    raise ValueError("impossible de déduire row_bytes (trajectoire vide)")


class ShuffledTransitionBatches:
    """Mélange borné fichier par fichier, réitérable par epoch."""

    def __init__(
        self,
        source: TrajectorySplitSource,
        *,
        batch_size: int,
        shuffle_buffer_mib: int,
        seed: int,
        epoch: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size invalide")
        if shuffle_buffer_mib < 1:
            raise ValueError("shuffle_buffer_mib invalide")
        self._source = source
        self._batch_size = batch_size
        self._budget_bytes = shuffle_buffer_mib * 1024**2
        self._seed = seed
        self._epoch = epoch
        self._row_bytes: int | None = None
        self._capacity = 0
        self._observation: np.ndarray | None = None
        self._action_mask: np.ndarray | None = None
        self._acts: np.ndarray | None = None
        self._buffer_size = 0
        self._out_obs: list[np.ndarray] = []
        self._out_mask: list[np.ndarray] = []
        self._out_acts: list[np.ndarray] = []

    def close(self) -> None:
        self._observation = None
        self._action_mask = None
        self._acts = None
        self._buffer_size = 0
        self._out_obs.clear()
        self._out_mask.clear()
        self._out_acts.clear()

    def __iter__(self) -> Iterator[TransitionBatch]:
        return self._run()

    def _run(self) -> Iterator[TransitionBatch]:
        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed, self._epoch, 1]),
        )
        file_perm = rng.permutation(len(self._source.files))
        transition_iter = self._iter_transitions(file_perm)
        try:
            first = next(transition_iter)
        except StopIteration:
            return
        self._init_buffers(first)
        obs_row, mask_row, act_row = first
        self._buffer_put(obs_row, mask_row, act_row)
        del first, obs_row, mask_row, act_row

        try:
            for obs_row, mask_row, act_row in transition_iter:
                if self._buffer_size < self._capacity:
                    self._buffer_put(obs_row, mask_row, act_row)
                else:
                    idx = int(rng.integers(0, self._capacity))
                    out_o, out_m, out_a = self._buffer_take(idx)
                    ready = self._push_out_row(out_o, out_m, out_a)
                    if ready is not None:
                        yield ready
                    self._buffer_write(idx, obs_row, mask_row, act_row)
                del obs_row, mask_row, act_row

            while self._buffer_size > 0:
                idx = int(rng.integers(0, self._buffer_size))
                out_o, out_m, out_a = self._buffer_take_drain(idx)
                ready = self._push_out_row(out_o, out_m, out_a)
                if ready is not None:
                    yield ready

            tail = self._flush_out_batch()
            if tail is not None:
                yield tail
        finally:
            self.close()

    def _iter_transitions(
        self,
        file_perm: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for file_idx in file_perm:
            path = self._source.files[int(file_idx)]
            traj = load_trajectory_pickle(path)
            try:
                for row in _transition_rows(traj):
                    yield row
            finally:
                del traj

    def _init_buffers(
        self,
        first_row: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        obs_row, mask_row, _ = first_row
        if self._row_bytes is None:
            self._row_bytes = int(
                obs_row.nbytes + mask_row.nbytes + 2 * np.dtype(np.int64).itemsize,
            )
            cap = self._budget_bytes // self._row_bytes
            if cap < 1:
                raise ValueError("une transition dépasse shuffle_buffer_mib")
            self._capacity = min(self._source.expected_transitions, cap)
        self._observation = np.empty(
            (self._capacity, *obs_row.shape),
            dtype=obs_row.dtype,
        )
        self._action_mask = np.empty(
            (self._capacity, *mask_row.shape),
            dtype=mask_row.dtype,
        )
        self._acts = np.empty((self._capacity, 2), dtype=np.int64)
        self._buffer_size = 0

    def _buffer_put(
        self,
        obs_row: np.ndarray,
        mask_row: np.ndarray,
        act_row: np.ndarray,
    ) -> None:
        assert self._observation is not None
        idx = self._buffer_size
        self._observation[idx] = obs_row
        assert self._action_mask is not None
        self._action_mask[idx] = mask_row
        assert self._acts is not None
        self._acts[idx] = act_row
        self._buffer_size += 1

    def _buffer_write(
        self,
        idx: int,
        obs_row: np.ndarray,
        mask_row: np.ndarray,
        act_row: np.ndarray,
    ) -> None:
        assert self._observation is not None and self._action_mask is not None
        assert self._acts is not None
        self._observation[idx] = obs_row
        self._action_mask[idx] = mask_row
        self._acts[idx] = act_row

    def _buffer_take(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._observation is not None and self._action_mask is not None
        assert self._acts is not None
        return (
            self._observation[idx].copy(),
            self._action_mask[idx].copy(),
            self._acts[idx].copy(),
        )

    def _buffer_take_drain(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._observation is not None and self._action_mask is not None
        assert self._acts is not None
        last = self._buffer_size - 1
        out = (
            self._observation[idx].copy(),
            self._action_mask[idx].copy(),
            self._acts[idx].copy(),
        )
        if idx != last:
            self._observation[idx] = self._observation[last]
            self._action_mask[idx] = self._action_mask[last]
            self._acts[idx] = self._acts[last]
        self._buffer_size = last
        return out

    def _push_out_row(
        self,
        obs_row: np.ndarray,
        mask_row: np.ndarray,
        act_row: np.ndarray,
    ) -> TransitionBatch | None:
        self._out_obs.append(obs_row)
        self._out_mask.append(mask_row)
        self._out_acts.append(act_row)
        if len(self._out_obs) >= self._batch_size:
            return self._flush_out_batch()
        return None

    def _flush_out_batch(self) -> TransitionBatch | None:
        if not self._out_obs:
            return None
        batch = _owned_batch(self._out_obs, self._out_mask, self._out_acts)
        self._out_obs = []
        self._out_mask = []
        self._out_acts = []
        return batch
