import numpy as np
import pytest
import torch
from imitation.data.types import DictObs, Trajectory
from vgc_bench.src.utils import act_len, chunk_obs_len

from pokecoach_trainer.compatibility import ModelContract
from pokecoach_trainer.errors import BlockingMaskError
from pokecoach_trainer.trajectory_dataset import (
    synthetic_bc_mask,
    trajectory_to_imitation,
    validate_trajectory_contract,
)


def _contract() -> ModelContract:
    return ModelContract(
        fingerprint="fp",
        vgc_bench_commit="x",
        imitation_commit="x",
        poke_env_commit="x",
        act_len=act_len,
        mask_rules={},
    )


def _traj(steps: int = 2) -> Trajectory:
    obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
    acts = np.zeros((steps, 2), dtype=np.int64)
    acts[:, 1] = 1
    return trajectory_to_imitation(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))


def test_len_obs_equals_len_acts_plus_one():
    validate_trajectory_contract(_traj(), _contract(), replay_id="r", player_side="p1")


def test_negative_action_index_blocks():
    traj = _traj()
    traj.acts[0, 0] = -1
    with pytest.raises(BlockingMaskError) as exc:
        validate_trajectory_contract(traj, _contract(), replay_id="r", player_side="p1")
    assert exc.value.details["replay_id"] == "r"


def test_conditional_mask_blocks_duplicate_switch():
    steps = 1
    obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
    mask = synthetic_bc_mask(steps + 1)
    dict_obs = DictObs({"observation": obs, "action_mask": mask})
    acts = np.array([[2, 2]], dtype=np.int64)
    traj = Trajectory(obs=dict_obs, acts=acts, infos=None, terminal=True)
    with pytest.raises(BlockingMaskError):
        validate_trajectory_contract(traj, _contract(), replay_id="r", player_side="p1")
