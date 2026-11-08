"""Tests CompleteBatchBC en mode flux de batches."""

from __future__ import annotations

import numpy as np
import pytest
from imitation.data.types import Trajectory

from pokecoach_trainer.bc_adapter import CompleteBatchBC
from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.training import build_ppo_container
from pokecoach_trainer.training_data import iter_transition_batches
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation
from vgc_bench.src.utils import chunk_obs_len


def _single_traj(n_transitions: int):
    obs = np.zeros((n_transitions + 1, 12 * chunk_obs_len), dtype=np.float32)
    acts = np.zeros((n_transitions, 2), dtype=np.int64)
    acts[:, 1] = 1
    return trajectory_to_imitation(
        Trajectory(obs=obs, acts=acts, infos=None, terminal=True),
    )


def _bc_streaming(ppo, n_transitions: int, batch_size: int = 32) -> CompleteBatchBC:
    traj = _single_traj(n_transitions)
    batches = iter_transition_batches([traj], batch_size=batch_size)
    bc = CompleteBatchBC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        batch_size=batch_size,
        minibatch_size=batch_size,
        device="cpu",
        optimizer_kwargs={"lr": 1e-3},
    )
    bc.set_transition_batches(
        batches,
        expected_transitions=n_transitions,
        batch_size=batch_size,
    )
    return bc


@pytest.mark.parametrize(
    "n_transitions,expected_steps",
    [(1, 1), (31, 1), (32, 1), (33, 2)],
)
def test_streaming_transition_counts(n_transitions: int, expected_steps: int):
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    bc = _bc_streaming(ppo, n_transitions)
    bc.train(n_epochs=1, progress_bar=False)
    assert bc.last_train_stats.transitions_seen == n_transitions
    assert bc.last_train_stats.optimizer_steps == expected_steps


def test_streaming_resets_counters_between_epochs():
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    for _ in range(2):
        bc = _bc_streaming(ppo, 33)
        bc.train(n_epochs=1, progress_bar=False)
        assert bc.last_train_stats.transitions_seen == 33
        assert bc.last_train_stats.optimizer_steps == 2


def test_streaming_rejects_wrong_transition_count():
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    traj = _single_traj(32)
    batches = iter_transition_batches([traj], batch_size=32)
    bc = CompleteBatchBC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        batch_size=32,
        minibatch_size=32,
        device="cpu",
        optimizer_kwargs={"lr": 1e-3},
    )
    bc.set_transition_batches(batches, expected_transitions=33, batch_size=32)
    with pytest.raises(ValueError, match="transitions observées"):
        bc.train(n_epochs=1, progress_bar=False)


def test_streaming_rejects_extra_transitions():
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    traj = _single_traj(34)
    batches = iter_transition_batches([traj], batch_size=32)
    bc = CompleteBatchBC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        batch_size=32,
        minibatch_size=32,
        device="cpu",
        optimizer_kwargs={"lr": 1e-3},
    )
    bc.set_transition_batches(batches, expected_transitions=33, batch_size=32)
    with pytest.raises(ValueError, match="transitions observées"):
        bc.train(n_epochs=1, progress_bar=False)


def test_streaming_rejects_multi_epoch():
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    bc = _bc_streaming(ppo, 8)
    with pytest.raises(ValueError, match="n_epochs"):
        bc.train(n_epochs=2, progress_bar=False)


def test_streaming_ignores_flatten_trajectories(monkeypatch):
    import imitation.data.rollout as rollout

    def _boom(*_args, **_kwargs):
        raise AssertionError("flatten_trajectories ne doit pas être appelé")

    monkeypatch.setattr(rollout, "flatten_trajectories", _boom)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    bc = _bc_streaming(ppo, 16)
    bc.train(n_epochs=1, progress_bar=False)
    assert bc._cached_trajectories is None


def _truncate_batches(batches, limit: int):
    for i, batch in enumerate(batches):
        if i >= limit:
            return
        yield batch


def test_streaming_truncated_loader_fails():
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    traj = _single_traj(64)
    full = list(iter_transition_batches([traj], batch_size=32))
    assert len(full) == 2
    bc = CompleteBatchBC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        batch_size=32,
        minibatch_size=32,
        device="cpu",
        optimizer_kwargs={"lr": 1e-3},
    )
    bc.set_transition_batches(_truncate_batches(full, 1), expected_transitions=64, batch_size=32)
    with pytest.raises(ValueError, match="transitions observées"):
        bc.train(n_epochs=1, progress_bar=False)
