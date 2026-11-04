"""Dataset PyTorch / imitation pour transitions BC."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from imitation.data.types import DictObs, Trajectory
from torch.utils.data import Dataset
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.utils import act_len

from pokecoach_trainer.compatibility import ModelContract
from pokecoach_trainer.errors import BlockingMaskError


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


def validate_trajectory_contract(
    traj: Trajectory,
    contract: ModelContract,
    *,
    replay_id: str,
    player_side: str,
) -> None:
    """
    Contrôle dimensions, bornes et masque conditionnel BC.

    Raises:
        BlockingMaskError: cible interdite bloquante pour toute la publication.
        ValueError: forme incohérente ou transition vide.
    """

    if len(traj.acts) == 0:
        raise ValueError("trajectoire sans transitions")
    obs = _obs_arrays(traj.obs)
    observation = obs.get("observation")
    if observation is None:
        raise ValueError("observation manquante")
    obs_arr = np.asarray(observation)
    if obs_arr.shape[0] != len(traj.acts) + 1:
        raise ValueError("len(obs) doit valoir len(acts)+1")
    if not np.isfinite(obs_arr).all():
        raise ValueError("observation non finie")

    acts = np.asarray(traj.acts)
    if acts.ndim != 2 or acts.shape[1] != 2:
        raise ValueError("actions attendues (steps, 2)")
    max_index = 2 * contract.act_len - 1
    for step, (a1, a2) in enumerate(acts):
        if a1 < 0 or a2 < 0 or a1 >= contract.act_len or a2 >= contract.act_len:
            raise BlockingMaskError(
                "action_index_out_of_bounds",
                details={
                    "replay_id": replay_id,
                    "player_side": player_side,
                    "step": step,
                    "actions": [int(a1), int(a2)],
                    "act_len": contract.act_len,
                },
            )

    mask_arr = obs.get("action_mask")
    if mask_arr is None:
        mask_arr = synthetic_bc_mask(len(traj.acts) + 1)
    mask_np = np.asarray(mask_arr, dtype=np.float32)
    for step, (a1, a2) in enumerate(acts):
        if mask_np[step, a1] == 0:
            raise BlockingMaskError(
                "blocking_mask_first_target",
                details={
                    "replay_id": replay_id,
                    "player_side": player_side,
                    "step": step,
                    "actions": [int(a1), int(a2)],
                    "mask_slice": mask_np[step].tolist(),
                },
            )
        mask_t = torch.as_tensor(mask_np[step : step + 1].copy())
        updated = MaskedActorCriticPolicy._update_mask(
            mask_t, torch.tensor([[int(a1)]])
        )
        if updated[0, act_len + int(a2)] == 0:
            raise BlockingMaskError(
                "blocking_mask_second_target",
                details={
                    "replay_id": replay_id,
                    "player_side": player_side,
                    "step": step,
                    "actions": [int(a1), int(a2)],
                    "mask_slice": updated[0].tolist(),
                },
            )
        if int(a2) >= max_index:
            raise BlockingMaskError(
                "action_index_out_of_bounds",
                details={
                    "replay_id": replay_id,
                    "player_side": player_side,
                    "step": step,
                    "actions": [int(a1), int(a2)],
                },
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
