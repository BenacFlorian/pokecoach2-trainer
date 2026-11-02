"""Dataset PyTorch / imitation pour transitions BC."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from imitation.data.types import DictObs, Trajectory
from torch.utils.data import Dataset
from vgc_bench.src.utils import act_len


def synthetic_bc_mask(n_steps: int) -> np.ndarray:
    """Masque BC amont : ones sauf plages mega/z/dmax/tera (47:87 par Pokémon)."""

    mask = np.ones((n_steps, 2 * act_len), dtype=np.float32)
    mask[:, 47:87] = 0
    mask[:, act_len + 47 : act_len + 87] = 0
    return mask


def _obs_arrays(obs) -> dict:
    if isinstance(obs, DictObs):
        return dict(obs._d)
    if isinstance(obs, dict):
        return obs
    raise TypeError(f"observation inattendue: {type(obs)}")


def trajectory_to_imitation(traj: Trajectory) -> Trajectory:
    obs = traj.obs
    if isinstance(obs, (dict, DictObs)):
        return traj
    n_obs = len(traj.acts) + 1
    action_mask = synthetic_bc_mask(n_obs)
    dict_obs = DictObs({"observation": obs, "action_mask": action_mask})
    return Trajectory(obs=dict_obs, acts=traj.acts, infos=traj.infos, terminal=traj.terminal)


def assert_actions_allowed(traj: Trajectory, replay_id: str) -> None:
    """Lève si une action humaine est interdite par le masque synthétique."""

    obs = _obs_arrays(traj.obs)
    mask = obs["action_mask"]
    acts = traj.acts
    for step, (a1, a2) in enumerate(acts):
        if mask[step, a1] == 0 or mask[step, act_len + a2] == 0:
            raise ValueError(
                f"action interdite par masque BC replay={replay_id} step={step} acts=({a1},{a2})"
            )


def load_trajectory_pickle(path: Path) -> Trajectory:
    with path.open("rb") as f:
        raw = pickle.load(f)
    traj = trajectory_to_imitation(raw)
    assert_actions_allowed(traj, path.stem)
    return traj


class TrajectoryFileDataset(Dataset):
    """Charge une liste de fichiers `.pkl` (une trajectoire par fichier)."""

    def __init__(self, files: list[Path]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Trajectory:
        return load_trajectory_pickle(self.files[idx])


def count_transitions(files: list[Path]) -> int:
    total = 0
    for p in files:
        traj = load_trajectory_pickle(p)
        total += len(traj.acts)
    return total
