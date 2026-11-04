import math

import numpy as np
import pytest
from imitation.data.types import Trajectory
from vgc_bench.src.utils import chunk_obs_len
from pokecoach_trainer.bc_adapter import CompleteBatchBC, count_transitions
from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.training import build_ppo_container
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation


def _trajectories_with_transitions(ppo, n_transitions: int):
    obs = np.zeros((n_transitions + 1, 12 * chunk_obs_len), dtype=np.float32)
    acts = np.zeros((n_transitions, 2), dtype=np.int64)
    acts[:, 1] = 1
    traj = trajectory_to_imitation(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))
    return [traj]


@pytest.mark.parametrize("n_transitions", [1, 31, 32, 33])
def test_complete_batch_bc_epochs_cover_all_transitions(n_transitions):
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    demos = _trajectories_with_transitions(ppo, n_transitions)
    rng = np.random.default_rng(0)
    batch_size = 32
    bc = CompleteBatchBC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=rng,
        policy=ppo.policy,
        batch_size=batch_size,
        minibatch_size=batch_size,
        device="cpu",
        optimizer_kwargs={"lr": 1e-3},
    )
    bc.remember_trajectories(demos)
    bc.set_demonstrations(demos)
    effective = min(batch_size, n_transitions)
    assert bc.batch_size == effective
    bc.train(n_epochs=1, progress_bar=False)
    assert bc.last_train_stats.transitions_seen == count_transitions(demos)
    expected_steps = max(1, math.ceil(n_transitions / effective))
    assert bc.last_train_stats.optimizer_steps == expected_steps


def test_n33_runs_two_optimizer_steps_before_train_returns():
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    demos = _trajectories_with_transitions(ppo, 33)
    bc = CompleteBatchBC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(1),
        policy=ppo.policy,
        batch_size=32,
        minibatch_size=32,
        device="cpu",
        optimizer_kwargs={"lr": 1e-3},
    )
    bc.remember_trajectories(demos)
    bc.set_demonstrations(demos)
    bc.train(n_epochs=1, progress_bar=False)
    assert bc.last_train_stats.optimizer_steps == 2
